import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from data_convert_refactored.config import ActionEncoding, StatsMode
from data_convert_refactored.preparation.dataset_stats import (
    effective_official_stats,
    prepare_dataset_stats,
    validate_official_stats,
)


class OfficialStatsTests(unittest.TestCase):
    def test_accepts_collect_stats_without_sidecar_or_optional_fields(self):
        stats = {
            "actions_min": [-1.0] * 14,
            "actions_max": [1.0] * 14,
            "proprio_min": [-2.0] * 16,
            "proprio_max": [2.0] * 16,
        }
        validate_official_stats(stats, action_dimension=14, proprio_dimension=16)
        self.assertEqual(stats["actions_min"].dtype, np.float32)

    def test_rejects_wrong_official_action_dimension(self):
        stats = {
            "actions_min": [-1.0] * 10,
            "actions_max": [1.0] * 10,
            "proprio_min": [-2.0] * 16,
            "proprio_max": [2.0] * 16,
        }
        with self.assertRaisesRegex(ValueError, r"shape \(14,\)"):
            validate_official_stats(stats, action_dimension=14, proprio_dimension=16)

    def test_extracts_legacy_fields_for_dual_arm_rotation6d(self):
        source = {
            "actions_min": list(range(14)),
            "actions_max": list(range(20, 34)),
            "proprio_min": [-2.0] * 16,
            "proprio_max": [2.0] * 16,
        }
        output = effective_official_stats(source, ActionEncoding.ROTATION_6D, 20)
        self.assertEqual(output["actions_min"][0:3].tolist(), [0, 1, 2])
        self.assertEqual(output["actions_min"][3:9].tolist(), [-1.0] * 6)
        self.assertEqual(output["actions_min"][9], 6)
        self.assertEqual(output["actions_min"][10:13].tolist(), [7, 8, 9])
        self.assertEqual(output["actions_min"][13:19].tolist(), [-1.0] * 6)
        self.assertEqual(output["actions_min"][19], 13)
        self.assertEqual(output["actions_max"][3:9].tolist(), [1.0] * 6)
        self.assertEqual(output["actions_max"][13:19].tolist(), [1.0] * 6)
        self.assertIs(output["proprio_min"], source["proprio_min"])
        self.assertEqual(len(source["actions_min"]), 14)

    def test_prepare_uses_official_loader_and_keeps_source_and_effective_stats(self):
        source = {
            "actions_min": [-1.0] * 14,
            "actions_max": [1.0] * 14,
            "proprio_min": [-2.0] * 16,
            "proprio_max": [2.0] * 16,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset_statistics.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            prepared = prepare_dataset_stats(
                path,
                StatsMode.OFFICIAL,
                ActionEncoding.ROTATION_6D,
                action_dimension=20,
            )
        self.assertEqual(prepared.source["actions_min"].shape, (14,))
        self.assertEqual(prepared.effective["actions_min"].shape, (20,))
        self.assertEqual(
            prepared.effective["action_stats_runtime_adapter"],
            "dual_arm_legacy_euler_14d_to_rotation6d_20d",
        )


if __name__ == "__main__":
    unittest.main()
