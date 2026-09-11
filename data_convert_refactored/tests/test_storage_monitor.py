from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data_convert_refactored.storage_monitor import StorageMonitor, _filesystem_snapshot


class StorageMonitorTest(unittest.TestCase):
    def test_filesystem_snapshot_has_capacity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = _filesystem_snapshot(Path(directory))
        self.assertGreater(snapshot["storage_total_bytes"], 0)
        self.assertGreaterEqual(snapshot["storage_available_bytes"], 0)
        self.assertIsNotNone(snapshot["storage_used_percent"])

    def test_probe_allocates_fsyncs_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            monitor = StorageMonitor(root, interval_s=60.0, probe_mib=1)
            monitor.local_path = root / "local.jsonl"
            monitor.probe("test_probe")
            self.assertFalse((root / f".cosmos_storage_probe_{__import__('os').getpid()}").exists())
            records = [json.loads(line) for line in monitor.local_path.read_text().splitlines()]
            self.assertEqual(records[-1]["probe_status"], "success")
            self.assertEqual(records[-1]["probe_bytes"], 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
