import json
import tempfile
import unittest
from pathlib import Path

from data_convert_refactored.preparation.episode import discover_episodes


class EpisodeManifestTests(unittest.TestCase):
    def test_manifest_freezes_subset_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for episode_id in ("episode_a", "episode_b", "episode_c"):
                path = root / episode_id / "data" / "trajectory.hdf5"
                path.parent.mkdir(parents=True)
                path.touch()
                paths.append(path.resolve())
            manifest = root / "selected.json"
            manifest.write_text(
                json.dumps([{"path": str(paths[2])}, {"path": str(paths[0])}]),
                encoding="utf-8",
            )

            selected = discover_episodes(root, episode_manifest=manifest)

        self.assertEqual(selected, [paths[2], paths[0]])

    def test_manifest_rejects_path_outside_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            input_root.mkdir()
            outside = root / "outside" / "data" / "trajectory.hdf5"
            outside.parent.mkdir(parents=True)
            outside.touch()
            manifest = root / "selected.json"
            manifest.write_text(json.dumps([str(outside)]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "outside --input"):
                discover_episodes(input_root, episode_manifest=manifest)


if __name__ == "__main__":
    unittest.main()
