#!/usr/bin/env python3
"""Keep Honcho embeddings on the best healthy Ollama endpoint.

The watchdog falls back from MBP2020 to local Pi Ollama after repeated remote
embedding failures. While local fallback is active, it periodically performs
bounded safe repairs of the MBP path and switches Honcho back only after the
actual remote embeddings endpoint passes consecutive checks.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import re
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HONCHO_DIR = Path(os.environ.get("HONCHO_DIR", "/home/pi/honcho"))
CONFIG_PATH = HONCHO_DIR / "config.toml"
STATE_PATH = Path(
    os.environ.get(
        "HONCHO_EMBED_WATCHDOG_STATE",
        "/home/pi/.cache/honcho-embedding-endpoint-watchdog.json",
    )
)
DIGEST_EVENTS_PATH = Path(
    os.environ.get(
        "HONCHO_EMBEDDING_DIGEST_EVENTS",
        "/home/pi/.hermes/data/automation-failure-repair/digest-events.jsonl",
    )
)

REMOTE_BASE_URL = os.environ.get(
    "HONCHO_REMOTE_EMBEDDING_BASE_URL", "http://172.18.0.1:11435/v1"
)
LOCAL_BASE_URL = os.environ.get(
    "HONCHO_LOCAL_EMBEDDING_BASE_URL", "http://host.docker.internal:11434/v1"
)
LOCAL_PROBE_BASE_URL = os.environ.get(
    "HONCHO_LOCAL_OLLAMA_BASE_URL", "http://localhost:11434"
)
MODEL = os.environ.get("HONCHO_EMBEDDING_MODEL", "mxbai-embed-large")
# The Pi fallback can legitimately take 30-45 seconds while Ollama loads or
# contends for memory. Keep the watchdog above that observed cold-path latency
# so a slow but healthy local provider is not misclassified as unavailable.
EMBEDDING_PROBE_TIMEOUT_SECONDS = float(
    os.environ.get("HONCHO_EMBEDDING_PROBE_TIMEOUT_SECONDS", "60")
)
REMOTE_FAILURE_THRESHOLD = int(os.environ.get("HONCHO_REMOTE_FAILURE_THRESHOLD", "3"))
REMOTE_RESTORE_SUCCESS_THRESHOLD = int(
    os.environ.get("HONCHO_REMOTE_RESTORE_SUCCESS_THRESHOLD", "3")
)
REMOTE_REPAIR_INTERVAL_SECONDS = int(
    float(os.environ.get("HONCHO_REMOTE_REPAIR_INTERVAL_MINUTES", "15")) * 60
)
REMOTE_AUTO_RESTORE = os.environ.get("HONCHO_REMOTE_AUTO_RESTORE", "1").lower() not in {
    "0",
    "false",
    "no",
}
REMOTE_PROXY_SERVICE = os.environ.get(
    "HONCHO_REMOTE_PROXY_SERVICE", "honcho-ollama-mbp2020-proxy.service"
)
REMOTE_SSH_TARGET = os.environ.get(
    "HONCHO_REMOTE_SSH_TARGET", "andylin@mbp2020.tail9e793a.ts.net"
)
REMOTE_OLLAMA_LAUNCH_AGENT = os.environ.get(
    "HONCHO_REMOTE_OLLAMA_LAUNCH_AGENT", "com.andy.ollama-lan"
)
EMAIL_TARGET = os.environ.get("HONCHO_EMBEDDING_ALERT_TARGET", "email:andylin@gmail.com")
EMAIL_SUBJECT = os.environ.get(
    "HONCHO_EMBEDDING_ALERT_SUBJECT", "[Hermes][Honcho] Embedding fallback"
)
NOTIFY_MODE = os.environ.get("HONCHO_EMBEDDING_NOTIFY_MODE", "digest").strip().lower()
HERMES_BIN = os.environ.get("HERMES_BIN", "/home/pi/.local/bin/hermes")

BASE_URL_RE = re.compile(
    r'(?m)^(\s*base_url\s*=\s*")(?P<url>[^"]+)("\s*)$', re.MULTILINE
)


def now() -> str:
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def load_state() -> dict[str, object]:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def save_state(state: dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(STATE_PATH)


def append_digest_event(event: dict[str, object]) -> None:
    """Record routine fallback/restore outcomes for the daily Supervisor digest."""
    DIGEST_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "incident_id": "honcho-embedding-endpoint-watchdog",
        "source": "honcho",
        "name": "Honcho embedding endpoint watchdog",
    }
    payload.update(event)
    with DIGEST_EVENTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def http_json(url: str, timeout: float = 4.0) -> tuple[bool, object | str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "honcho-embed-watchdog/1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(200_000)
            if resp.status != 200:
                return False, f"HTTP {resp.status}"
            return True, json.loads(body.decode("utf-8"))
    except (
        urllib.error.URLError,
        ConnectionError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def probe_ollama(base: str) -> tuple[bool, str]:
    ok, payload = http_json(f"{base.rstrip('/')}/api/tags")
    if not ok:
        return False, str(payload)
    models = []
    if isinstance(payload, dict):
        models = [str(m.get("name") or m.get("model") or "") for m in payload.get("models", [])]
    model_ok = any(m == MODEL or m == f"{MODEL}:latest" or m.startswith(f"{MODEL}:") for m in models)
    if not model_ok:
        return False, f"{MODEL} not present; models={models!r}"
    return True, f"ok models={models!r}"


def probe_openai_embeddings(base: str) -> tuple[bool, str]:
    """Probe the actual endpoint Honcho uses, not just Ollama /api/tags.

    The SSH tunnel can answer /api/tags while the OpenAI-compatible embeddings
    request hangs or fails. That false positive left Honcho configured to a dead
    provider, so this watchdog must validate /v1/embeddings directly.
    """
    url = f"{base.rstrip('/')}/embeddings"
    payload = json.dumps({"model": MODEL, "input": "honcho watchdog probe"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "honcho-embed-watchdog/1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=EMBEDDING_PROBE_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read(500_000).decode("utf-8"))
    except (
        urllib.error.URLError,
        ConnectionError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    try:
        dim = len(body["data"][0]["embedding"])
    except (KeyError, IndexError, TypeError) as exc:
        return False, f"malformed embedding response: {type(exc).__name__}: {exc}"
    if dim <= 0:
        return False, "empty embedding vector"
    return True, f"ok embedding_dim={dim}"


def read_current_base_url() -> str:
    text = CONFIG_PATH.read_text()
    in_embedding_override = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[embedding.MODEL_CONFIG.overrides]":
            in_embedding_override = True
            continue
        if in_embedding_override and stripped.startswith("["):
            break
        if in_embedding_override:
            match = re.match(r'base_url\s*=\s*"([^"]+)"', stripped)
            if match:
                return match.group(1)
    raise RuntimeError(f"Could not find embedding override base_url in {CONFIG_PATH}")


def replace_embedding_base_url(new_url: str) -> bool:
    text = CONFIG_PATH.read_text()
    lines = text.splitlines(keepends=True)
    in_embedding_override = False
    changed = False
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "[embedding.MODEL_CONFIG.overrides]":
            in_embedding_override = True
            out.append(line)
            continue
        if in_embedding_override and stripped.startswith("["):
            in_embedding_override = False
        if in_embedding_override and re.match(r"\s*base_url\s*=", line):
            indent = line[: len(line) - len(line.lstrip())]
            old = re.search(r'"([^"]+)"', line)
            if old and old.group(1) == new_url:
                out.append(line)
            else:
                out.append(f'{indent}base_url = "{new_url}"\n')
                changed = True
            continue
        out.append(line)
    if changed:
        CONFIG_PATH.write_text("".join(out))
    return changed


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        partial_output = exc.stdout or ""
        if isinstance(partial_output, bytes):
            partial_output = partial_output.decode(errors="replace")
        detail = partial_output.strip()
        message = f"command timed out after {timeout}s"
        return 124, f"{message}\n{detail}" if detail else message
    return proc.returncode, proc.stdout.strip()


def restart_honcho() -> str:
    code, output = run(["docker", "compose", "restart", "api", "deriver"], cwd=HONCHO_DIR, timeout=180)
    if code != 0:
        raise RuntimeError(f"docker compose restart failed ({code}):\n{output}")
    return output


def repair_remote_ollama_endpoint() -> str:
    """Best-effort repair of the MBP Ollama path before falling back locally."""
    repair_steps: list[str] = []

    if REMOTE_PROXY_SERVICE:
        code, output = run(["sudo", "-n", "systemctl", "restart", REMOTE_PROXY_SERVICE], timeout=30)
        repair_steps.append(
            f"sudo -n systemctl restart {REMOTE_PROXY_SERVICE}: rc={code}\n{output}"
        )

    if REMOTE_SSH_TARGET:
        remote_cmd = f'''
set -eu
uid=$(id -u)
agent={REMOTE_OLLAMA_LAUNCH_AGENT!r}
launchctl kickstart -k "gui/$uid/$agent" 2>/dev/null || true
for i in 1 2 3 4 5; do
  if curl -fsS --connect-timeout 2 --max-time 8 http://127.0.0.1:11434/api/tags >/dev/null; then
    echo "mbp_ollama_local_ok attempt=$i"
    exit 0
  fi
  sleep 2
done
echo "mbp_ollama_local_failed"
exit 1
'''
        code, output = run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                REMOTE_SSH_TARGET,
                remote_cmd,
            ],
            timeout=45,
        )
        repair_steps.append(
            f"ssh {REMOTE_SSH_TARGET} ollama kickstart: rc={code}\n{output}"
        )

    return "\n\n".join(repair_steps)


def verify_container_embedding() -> str:
    snippet = r'''
import asyncio, math
from src.config import settings, resolve_embedding_model_config
from src.embedding_client import _EmbeddingClient
cfg = resolve_embedding_model_config(settings.EMBEDDING.MODEL_CONFIG)
print("resolved_base_url=", cfg.base_url)
client = _EmbeddingClient(
    cfg,
    vector_dimensions=settings.EMBEDDING.VECTOR_DIMENSIONS,
    max_input_tokens=settings.EMBEDDING.MAX_INPUT_TOKENS,
    max_tokens_per_request=settings.EMBEDDING.MAX_TOKENS_PER_REQUEST,
    send_dimensions=False,
)
async def main():
    emb = await client.embed("Honcho local fallback watchdog verification")
    print("embedding_dim=", len(emb))
    print("has_nan=", any(math.isnan(x) for x in emb))
asyncio.run(main())
'''
    code, output = run(["docker", "exec", "-i", "honcho-api-1", "python", "-c", snippet], timeout=90)
    if code != 0:
        raise RuntimeError(f"container embedding verification failed ({code}):\n{output}")
    return output


def send_email(body: str, *, subject: str = EMAIL_SUBJECT) -> str:
    if not Path(HERMES_BIN).exists():
        return f"hermes binary missing at {HERMES_BIN}; email not sent"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
        fh.write(body)
        path = fh.name
    try:
        code, output = run(
            [HERMES_BIN, "send", "--quiet", "--to", EMAIL_TARGET, "--subject", subject, "--file", path],
            timeout=60,
        )
        if code != 0:
            return f"email send failed ({code}): {output}"
        return f"email sent to {EMAIL_TARGET} subject={subject!r}"
    finally:
        with contextlib.suppress(FileNotFoundError):
            Path(path).unlink()


def routine_notification(body: str, *, subject: str, event: dict[str, object]) -> str:
    """Notify for successful self-healing without spamming Andy.

    Fallback-to-local and restore-to-MBP are routine successful repairs. By
    default they go to the daily Supervisor digest. Set
    HONCHO_EMBEDDING_NOTIFY_MODE=immediate only while debugging the watchdog.
    """
    append_digest_event(event)
    if NOTIFY_MODE in {"immediate", "email", "all"}:
        return send_email(body, subject=subject)
    return f"digest event recorded at {DIGEST_EVENTS_PATH}"


def build_fallback_body(
    *, old_url: str, remote_reason: str, local_probe: str, restart_output: str, verify_output: str
) -> str:
    return f"""## Honcho embedding fallback activated

Honcho was configured to use MBP2020 for embeddings, but the remote Ollama endpoint failed. I switched Honcho back to local Pi Ollama and restarted the Honcho API/deriver containers.

- **Host:** {socket.gethostname()}
- **Time:** {now()}
- **Old endpoint:** `{old_url}`
- **New endpoint:** `{LOCAL_BASE_URL}`
- **Remote probe failure:** `{remote_reason}`
- **Local probe:** `{local_probe}`
- **Model:** `{MODEL}`

### Restart output
```text
{restart_output}
```

### Verification output
```text
{verify_output}
```

### Reference
`REF: HERMES-NOTIFY:honcho:embedding-fallback:mbp2020-offline`
"""


def build_restore_body(
    *, old_url: str, remote_reason: str, restart_output: str, verify_output: str, repair_output: str
) -> str:
    return f"""## Honcho embedding restored to MBP2020

The fallback watchdog confirmed MBP2020's actual embedding endpoint is healthy for {REMOTE_RESTORE_SUCCESS_THRESHOLD} consecutive checks. I switched Honcho back to the MBP2020 Ollama bridge and restarted the Honcho API/deriver containers.

- **Host:** {socket.gethostname()}
- **Time:** {now()}
- **Old endpoint:** `{old_url}`
- **New endpoint:** `{REMOTE_BASE_URL}`
- **Remote probe:** `{remote_reason}`
- **Model:** `{MODEL}`

### Repair output
```text
{repair_output or 'no repair needed'}
```

### Restart output
```text
{restart_output}
```

### Verification output
```text
{verify_output}
```

### Reference
`REF: HERMES-NOTIFY:honcho:embedding-restore:mbp2020-online`
"""


def should_repair_remote_from_local(state: dict[str, object]) -> bool:
    raw = state.get("last_remote_repair_attempt_epoch", 0)
    try:
        last_attempt = float(raw) if isinstance(raw, int | float | str) else 0.0
    except (TypeError, ValueError):
        last_attempt = 0.0
    return (time.time() - last_attempt) >= REMOTE_REPAIR_INTERVAL_SECONDS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="probe and report without changing config")
    parser.add_argument("--force", action="store_true", help="force fallback even if remote probe passes")
    parser.add_argument("--restore-now", action="store_true", help="switch local fallback back to remote after one passing remote probe")
    args = parser.parse_args()

    state = load_state()
    current_url = read_current_base_url()
    current_is_local = current_url == LOCAL_BASE_URL and not args.force
    remote_probe_base = REMOTE_BASE_URL.removesuffix("/v1")
    remote_ok, remote_tags_reason = probe_ollama(remote_probe_base)
    if remote_ok:
        remote_ok, remote_reason = probe_openai_embeddings(REMOTE_BASE_URL)
        if remote_ok:
            remote_reason = f"{remote_tags_reason}; {remote_reason}"
        else:
            remote_reason = f"tags ok ({remote_tags_reason}); embeddings failed: {remote_reason}"
    else:
        remote_reason = remote_tags_reason
    # Only prove the local endpoint when it is a candidate fallback target.
    # Probing it while Honcho already runs locally adds an expensive embedding
    # request every two minutes and can worsen the very queue saturation the
    # watchdog is meant to survive. Honcho's own health check covers the active
    # provider; the watchdog verifies local embeddings immediately before a
    # remote-to-local switch.
    should_probe_local_fallback = not current_is_local and (args.force or not remote_ok)
    if should_probe_local_fallback:
        local_ok, local_tags_reason = probe_ollama(LOCAL_PROBE_BASE_URL)
        if local_ok:
            local_openai_probe_base = f"{LOCAL_PROBE_BASE_URL.rstrip('/')}/v1"
            local_ok, local_reason = probe_openai_embeddings(local_openai_probe_base)
            if local_ok:
                local_reason = f"{local_tags_reason}; {local_reason}"
            else:
                local_reason = f"tags ok ({local_tags_reason}); embeddings failed: {local_reason}"
        else:
            local_reason = local_tags_reason
    else:
        local_ok = True
        local_reason = "probe skipped; local endpoint is active or not needed as fallback"
    raw_previous_failures = state.get("remote_failure_count", 0)
    previous_failures = int(raw_previous_failures) if isinstance(raw_previous_failures, int | str) else 0
    candidate_failure_count = 0 if remote_ok else previous_failures + 1
    repair_output = ""
    repaired_remote = False

    # Under embedding load one probe can miss its deadline while the endpoint
    # remains healthy. Restarting the shared SSH tunnel on the first miss drops
    # every in-flight Honcho and GBrain request. Repair only after the same
    # consecutive-failure threshold that authorizes fallback.
    repair_threshold_reached = candidate_failure_count >= REMOTE_FAILURE_THRESHOLD

    # If local fallback is active, periodically try to heal the MBP path and
    # automatically restore only after repeated successful real embedding probes.
    if (
        current_is_local
        and not remote_ok
        and repair_threshold_reached
        and not args.dry_run
        and should_repair_remote_from_local(state)
    ):
        state["last_remote_repair_attempt_epoch"] = time.time()
        state["last_remote_repair_attempt"] = now()
        repair_output = repair_remote_ollama_endpoint()
        repaired_ok, repaired_tags_reason = probe_ollama(remote_probe_base)
        if repaired_ok:
            repaired_ok, repaired_reason = probe_openai_embeddings(REMOTE_BASE_URL)
            if repaired_ok:
                repaired_reason = f"{repaired_tags_reason}; {repaired_reason}"
            else:
                repaired_reason = f"tags ok ({repaired_tags_reason}); embeddings failed: {repaired_reason}"
        else:
            repaired_reason = repaired_tags_reason
        if repaired_ok:
            remote_ok = True
            remote_reason = f"repaired remote endpoint; {repaired_reason}"
            repaired_remote = True
        else:
            remote_reason = f"{remote_reason}; repair attempted but still failing: {repaired_reason}"

    if (
        not remote_ok
        and repair_threshold_reached
        and not args.dry_run
        and not current_is_local
    ):
        repair_output = repair_remote_ollama_endpoint()
        repaired_ok, repaired_tags_reason = probe_ollama(remote_probe_base)
        if repaired_ok:
            repaired_ok, repaired_reason = probe_openai_embeddings(REMOTE_BASE_URL)
            if repaired_ok:
                repaired_reason = f"{repaired_tags_reason}; {repaired_reason}"
            else:
                repaired_reason = f"tags ok ({repaired_tags_reason}); embeddings failed: {repaired_reason}"
        else:
            repaired_reason = repaired_tags_reason
        if repaired_ok:
            remote_ok = True
            remote_reason = f"repaired remote endpoint; {repaired_reason}"
            repaired_remote = True
        else:
            remote_reason = f"{remote_reason}; repair attempted but still failing: {repaired_reason}"

    remote_failure_count = 0 if remote_ok else previous_failures + 1
    raw_previous_successes = state.get("remote_success_count", 0)
    previous_successes = int(raw_previous_successes) if isinstance(raw_previous_successes, int | str) else 0
    remote_success_count = previous_successes + 1 if remote_ok else 0

    summary = {
        "time": now(),
        "current_url": current_url,
        "remote_ok": remote_ok,
        "remote_reason": remote_reason,
        "local_ok": local_ok,
        "local_reason": local_reason,
        "remote_failure_threshold": REMOTE_FAILURE_THRESHOLD,
        "remote_restore_success_threshold": REMOTE_RESTORE_SUCCESS_THRESHOLD,
        "remote_failure_count": remote_failure_count,
        "remote_success_count": remote_success_count,
        "remote_auto_restore": REMOTE_AUTO_RESTORE,
        "remote_repair_interval_seconds": REMOTE_REPAIR_INTERVAL_SECONDS,
        "repaired_remote": repaired_remote,
        "repair_output": repair_output,
    }

    if current_is_local:
        should_restore = (
            REMOTE_AUTO_RESTORE
            and remote_ok
            and (args.restore_now or remote_success_count >= REMOTE_RESTORE_SUCCESS_THRESHOLD)
        )
        if args.dry_run:
            action = "would_switch_to" if should_restore else "would_stay_on"
            target = REMOTE_BASE_URL if should_restore else LOCAL_BASE_URL
            print(json.dumps(summary | {action: target}, indent=2))
            return 0
        if not should_restore:
            state.update(summary | {"last_status": "already_local"})
            save_state(state)
            return 0

        changed = replace_embedding_base_url(REMOTE_BASE_URL)
        restart_output = restart_honcho() if changed else "config already remote; restart skipped"
        try:
            verify_output = verify_container_embedding()
        except Exception as exc:
            rollback_changed = replace_embedding_base_url(LOCAL_BASE_URL)
            rollback_output = restart_honcho() if rollback_changed else "rollback skipped; config already local"
            rollback_verify = verify_container_embedding()
            state.update(
                summary
                | {
                    "last_status": "restore_verification_failed_rolled_back",
                    "restore_error": str(exc),
                    "rollback_output": rollback_output,
                    "rollback_verify_output": rollback_verify,
                }
            )
            save_state(state)
            append_digest_event(
                {
                    "event": "supervisor_triage_completed",
                    "status": "restore_verification_failed_rolled_back",
                    "outcome": "blocked",
                    "notification": "immediate_email",
                    "reason": f"Remote restore verification failed and watchdog rolled back to local: {exc}",
                    "repair": "Rolled Honcho embedding base_url back to local fallback and restarted Honcho api/deriver.",
                    "verification": rollback_verify[:1000],
                }
            )
            email_result = send_email(
                "Honcho embedding restore BLOCKED: remote verification failed; "
                f"rolled back to local.\n\nError:\n{exc}\n\nRollback verification:\n{rollback_verify}",
                subject="[Hermes][Honcho] Embedding restore blocked",
            )
            print(
                "Honcho embedding restore BLOCKED: remote verification failed; "
                f"rolled back to local. error={exc}; {email_result}"
            )
            return 2
        body = build_restore_body(
            old_url=current_url,
            remote_reason=str(remote_reason),
            restart_output=restart_output,
            verify_output=verify_output,
            repair_output=repair_output,
        )
        notify_result = routine_notification(
            body,
            subject="[Hermes][Honcho] Embedding restored to MBP2020",
            event={
                "event": "supervisor_auto_repair_completed",
                "status": "restored_to_remote",
                "outcome": "silent_digest",
                "notification": "silent_digest_ledger_only",
                "reason": f"Honcho embeddings restored to MBP2020 after {remote_success_count} consecutive healthy remote checks.",
                "repair": f"Switched embedding base_url from {current_url} to {REMOTE_BASE_URL}; restarted Honcho api/deriver.",
                "verification": verify_output[:1000],
            },
        )
        state.update(
            summary
            | {
                "last_status": "restored_to_remote",
                "restored_at": now(),
                "notification_result": notify_result,
                "remote_success_count": 0,
                "verify_output": verify_output,
            }
        )
        save_state(state)
        print(f"Honcho embedding restored: {current_url} -> {REMOTE_BASE_URL}; {notify_result}")
        return 0

    if current_url != REMOTE_BASE_URL and not args.force:
        state.update(summary | {"last_status": "unmanaged_endpoint"})
        save_state(state)
        print(f"[SILENT] unmanaged endpoint {current_url}; not changing")
        return 0

    if remote_ok and not args.force:
        state.update(summary | {"last_status": "remote_ok", "remote_failure_count": 0})
        save_state(state)
        return 0

    if not args.force and remote_failure_count < REMOTE_FAILURE_THRESHOLD:
        state.update(summary | {"last_status": "remote_probe_failed_waiting"})
        save_state(state)
        print(
            "[SILENT] remote probe failed "
            f"{remote_failure_count}/{REMOTE_FAILURE_THRESHOLD}: {remote_reason}"
        )
        return 0

    if not local_ok:
        state.update(summary | {"last_status": "blocked_local_unhealthy"})
        save_state(state)
        append_digest_event(
            {
                "event": "supervisor_triage_completed",
                "status": "fallback_blocked_local_unhealthy",
                "outcome": "blocked",
                "notification": "immediate_email",
                "reason": f"Remote failed ({remote_reason}); local fallback also failed ({local_reason}).",
                "repair": "No config change applied because fallback target is unhealthy.",
            }
        )
        email_result = send_email(
            "Honcho embedding fallback BLOCKED: remote and local embedding providers failed.\n\n"
            f"Remote: {remote_reason}\n\nLocal: {local_reason}",
            subject="[Hermes][Honcho] Embedding fallback blocked",
        )
        print(
            f"Honcho embedding fallback BLOCKED: remote failed ({remote_reason}); "
            f"local failed ({local_reason}); {email_result}"
        )
        return 2

    if args.dry_run:
        print(json.dumps(summary | {"would_switch_to": LOCAL_BASE_URL}, indent=2))
        return 0

    changed = replace_embedding_base_url(LOCAL_BASE_URL)
    restart_output = restart_honcho() if changed or args.force else "config already local; restart skipped"
    verify_output = verify_container_embedding()
    body = build_fallback_body(
        old_url=current_url,
        remote_reason=str(remote_reason),
        local_probe=str(local_reason),
        restart_output=restart_output,
        verify_output=verify_output,
    )
    notify_result = routine_notification(
        body,
        subject=EMAIL_SUBJECT,
        event={
            "event": "supervisor_auto_repair_completed",
            "status": "switched_to_local",
            "outcome": "silent_digest",
            "notification": "silent_digest_ledger_only",
            "reason": f"Honcho remote embeddings failed {remote_failure_count}/{REMOTE_FAILURE_THRESHOLD}; local fallback verified healthy.",
            "repair": f"Switched embedding base_url from {current_url} to {LOCAL_BASE_URL}; restarted Honcho api/deriver.",
            "verification": verify_output[:1000],
        },
    )
    state.update(
        summary
        | {
            "last_status": "switched_to_local",
            "switched_at": now(),
            "notification_result": notify_result,
            "verify_output": verify_output,
        }
    )
    save_state(state)
    print(f"Honcho embedding fallback activated: {current_url} -> {LOCAL_BASE_URL}; {notify_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
