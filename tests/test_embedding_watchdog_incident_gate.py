import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "honcho_embedding_endpoint_watchdog.py"
spec = importlib.util.spec_from_file_location("honcho_embedding_watchdog", MODULE_PATH)
assert spec is not None and spec.loader is not None
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class IncidentGateTests(unittest.TestCase):
    def test_shared_probe_lock_prevents_concurrent_watchdogs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.lock"
            first = watchdog.acquire_probe_lock(path)
            self.assertIsNotNone(first)
            second = watchdog.acquire_probe_lock(path)
            self.assertIsNone(second)
            first.close()
            third = watchdog.acquire_probe_lock(path)
            self.assertIsNotNone(third)
            third.close()

    def test_sustained_outage_requires_elapsed_grace_and_repair_attempt(self):
        state = {}
        failures, age = watchdog.update_outage_state(state, remote_ok=False, now_epoch=100.0)
        self.assertEqual((failures, age), (1, 0.0))
        self.assertFalse(
            watchdog.incident_ready_to_escalate(
                state,
                failure_count=3,
                now_epoch=1899.0,
                failure_threshold=3,
                grace_seconds=1800,
            )
        )
        state["repair_attempted_for_incident"] = True
        self.assertTrue(
            watchdog.incident_ready_to_escalate(
                state,
                failure_count=3,
                now_epoch=1900.0,
                failure_threshold=3,
                grace_seconds=1800,
            )
        )

    def test_one_alert_per_unrecovered_incident(self):
        state = {"first_remote_failure_epoch": 100.0}
        self.assertTrue(watchdog.claim_incident_alert(state, now_epoch=2000.0))
        self.assertFalse(watchdog.claim_incident_alert(state, now_epoch=2100.0))

    def test_new_outage_clears_prior_runtime_reload_latch(self):
        state = {"remote_runtime_reload_attempted_for_incident": True}
        failures, age = watchdog.update_outage_state(
            state, remote_ok=False, now_epoch=100.0
        )
        self.assertEqual((failures, age), (1, 0.0))
        self.assertNotIn("remote_runtime_reload_attempted_for_incident", state)

    def test_recovery_clears_outage_and_alert_latch(self):
        state = {
            "remote_failure_count": 9,
            "first_remote_failure_epoch": 100.0,
            "repair_attempted_for_incident": True,
            "incident_alerted": True,
            "incident_alerted_epoch": 2000.0,
        }
        failures, age = watchdog.update_outage_state(state, remote_ok=True, now_epoch=2200.0)
        self.assertEqual((failures, age), (0, 0.0))
        for key in (
            "first_remote_failure_epoch",
            "repair_attempted_for_incident",
            "incident_alerted",
            "incident_alerted_epoch",
        ):
            self.assertNotIn(key, state)

    def test_remote_only_recovery_reloads_runtime_after_repair_attempt(self):
        state = {"repair_attempted_for_incident": True}
        self.assertTrue(
            watchdog.should_reload_remote_runtime_after_recovery(
                state,
                current_url="http://remote.test/v1",
                remote_ok=True,
                current_is_local=False,
                remote_url="http://remote.test/v1",
            )
        )

    def test_remote_runtime_reload_is_incident_scoped(self):
        cases = (
            ({}, "http://remote.test/v1", True, False),
            (
                {
                    "repair_attempted_for_incident": True,
                    "remote_runtime_reload_attempted_for_incident": True,
                },
                "http://remote.test/v1",
                True,
                False,
            ),
            ({"repair_attempted_for_incident": True}, "http://remote.test/v1", False, False),
            ({"repair_attempted_for_incident": True}, "http://local.test/v1", True, True),
            ({"repair_attempted_for_incident": True}, "http://other.test/v1", True, False),
        )
        for state, current_url, remote_ok, current_is_local in cases:
            with self.subTest(
                state=state,
                current_url=current_url,
                remote_ok=remote_ok,
                current_is_local=current_is_local,
            ):
                self.assertFalse(
                    watchdog.should_reload_remote_runtime_after_recovery(
                        state,
                        current_url=current_url,
                        remote_ok=remote_ok,
                        current_is_local=current_is_local,
                        remote_url="http://remote.test/v1",
                    )
                )


if __name__ == "__main__":
    unittest.main()
