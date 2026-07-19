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


if __name__ == "__main__":
    unittest.main()
