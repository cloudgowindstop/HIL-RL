import json
import tempfile
import unittest
from pathlib import Path

from data_convert_refactored.conditioning.camera import CameraState
from data_convert_refactored.config import (
    ActionEncoding,
    ActionSource,
    ConversionConfig,
    CosmosRuntimeConfig,
    StatsMode,
)
from data_convert_refactored.cosmos_backend import CosmosBackend
from data_convert_refactored.conditioning.episode_labeling import EpisodeOutcome


class RuntimeConfigTests(unittest.TestCase):
    def test_runtime_config_resolves_camera_profile_and_chunk_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            stats = root / "stats.json"
            stats.write_text("{}", encoding="utf-8")
            cosmos = root / "cosmos.json"
            cosmos.write_text(json.dumps({"chunk_size": 24}), encoding="utf-8")
            config = ConversionConfig(
                input_dir=input_dir,
                output_dir=root / "output",
                task="test task",
                episode_outcome=EpisodeOutcome.SUCCESS,
                stats_mode=StatsMode.OFFICIAL,
                official_dataset_stats=stats,
                action_encoding=ActionEncoding.ROTATION_6D,
                action_source=ActionSource.PUPPET_NEXT_FRAME,
                skip_t5=True,
                cosmos_config_path=cosmos,
                camera_state=CameraState.NORMAL,
                save_clean_restore_latent=True,
            )
            runtime = CosmosBackend(config)._runtime_config()

        self.assertIsInstance(runtime, CosmosRuntimeConfig)
        self.assertEqual(runtime.chunk_size, 24)
        self.assertEqual(runtime.wrist_crop_fraction, 0.75)
        self.assertEqual(runtime.wrist_left_center_offset_x, 64)
        self.assertEqual(runtime.head_crop_top_pixels, 100)
        self.assertIsInstance(runtime.dataset_stats_path, Path)
        self.assertTrue(runtime.save_clean_restore_latent)


if __name__ == "__main__":
    unittest.main()
