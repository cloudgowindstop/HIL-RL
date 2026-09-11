from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from data_convert_refactored.resume import (
    begin_episode_transaction,
    inspect_committed_output,
    open_dataset_for_append,
    prepare_resume,
)


class ResumeIntegrityTest(unittest.TestCase):
    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

    def test_empty_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(inspect_committed_output(Path(directory)), (0, 0))

    def test_rejects_inconsistent_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text(
                json.dumps({"total_episodes": 1, "total_frames": 5}), encoding="utf-8"
            )
            self._write_jsonl(
                root / "meta/episodes.jsonl",
                [{"episode_index": 0, "length": 5, "tasks": ["test"]}],
            )
            self._write_jsonl(root / "meta/episodes_stats.jsonl", [])
            with self.assertRaisesRegex(RuntimeError, "inconsistent resume tail"):
                inspect_committed_output(root)

    def test_removes_only_zero_byte_next_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "data/chunk-000").mkdir(parents=True)
            (root / "meta/info.json").write_text(
                json.dumps({"total_episodes": 0, "total_frames": 0}), encoding="utf-8"
            )
            orphan = root / "data/chunk-000/episode_000000.parquet"
            orphan.touch()
            self.assertEqual(inspect_committed_output(root), (0, 0))
            self.assertFalse(orphan.exists())


class AppendWriterTest(unittest.TestCase):
    def _config(self, base: Path, output: Path):
        from data_convert_refactored.conditioning.episode_labeling import EpisodeOutcome
        from data_convert_refactored.config import ConversionConfig, StatsMode

        stats = base / "stats.json"
        cosmos = base / "cosmos.json"
        stats.write_text("{}", encoding="utf-8")
        cosmos.write_text("{}", encoding="utf-8")
        return ConversionConfig(
            input_dir=base,
            output_dir=output,
            task="test",
            episode_outcome=EpisodeOutcome.SUCCESS,
            stats_mode=StatsMode.OFFICIAL,
            official_dataset_stats=stats,
            skip_t5=True,
            cosmos_config_path=cosmos,
        )

    def test_append_writer_does_not_load_old_rows(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = {"value": {"dtype": "float32", "shape": (1,), "names": None}}
            dataset = LeRobotDataset.create(
                "resume_test", fps=16, features=features, root=root, use_videos=False
            )
            dataset.add_frame({"value": np.array([1], dtype=np.float32)}, task="test")
            dataset.save_episode()

            resumed = open_dataset_for_append(root, "resume_test")
            self.assertEqual(resumed.meta.total_episodes, 1)
            self.assertEqual(len(resumed.hf_dataset), 0)
            self.assertEqual(resumed.episode_buffer["episode_index"], 1)
            resumed.add_frame({"value": np.array([2], dtype=np.float32)}, task="test")
            resumed.save_episode()

            self.assertEqual(resumed.meta.total_episodes, 2)
            self.assertTrue((root / "data/chunk-000/episode_000001.parquet").is_file())
            self.assertEqual(inspect_committed_output(root), (2, 2))

    def test_interrupted_metadata_save_rolls_back_to_last_commit(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = {"value": {"dtype": "float32", "shape": (1,), "names": None}}
            dataset = LeRobotDataset.create(
                "resume_test", fps=16, features=features, root=root, use_videos=False
            )
            dataset.add_frame({"value": np.array([1], dtype=np.float32)}, task="test")
            dataset.save_episode()
            begin_episode_transaction(root, 1, root / "source.hdf5")

            info_path = root / "meta/info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["total_episodes"] = 2
            info["total_frames"] = 2
            info_path.write_text(json.dumps(info), encoding="utf-8")
            with (root / "meta/episodes.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps({"episode_index": 1, "length": 1, "tasks": ["test"]}) + "\n")
            orphan = root / "data/chunk-000/episode_000001.parquet"
            orphan.write_bytes(b"PAR1incompletePAR1")

            self.assertEqual(inspect_committed_output(root), (1, 1))
            self.assertFalse(orphan.exists())
            restored = json.loads(info_path.read_text(encoding="utf-8"))
            self.assertEqual(restored["total_episodes"], 1)

    def test_resume_manifest_rejects_semantic_change(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / "output"
            source = base / "episode/data/trajectory.hdf5"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"source")
            dataset = LeRobotDataset.create(
                "resume_test",
                fps=16,
                features={"value": {"dtype": "float32", "shape": (1,), "names": None}},
                root=output,
                use_videos=False,
            )
            dataset.add_frame({"value": np.array([1], dtype=np.float32)}, task="test")
            dataset.save_episode()
            config = self._config(base, output)

            state = prepare_resume(config, [source], enabled=True)
            self.assertEqual(state.start_episode, 1)
            with self.assertRaisesRegex(RuntimeError, "manifest mismatch"):
                prepare_resume(replace(config, task="changed"), [source], enabled=True)


if __name__ == "__main__":
    unittest.main()
