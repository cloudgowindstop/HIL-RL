import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_convert_refactored.tools.compare_conversion_outputs import compare_datasets


class CompareConversionOutputsTest(unittest.TestCase):
    def write_episode(self, root: Path, video_values, task_values=(0, 0)) -> None:
        output = root / "data" / "chunk-000" / "episode_000000.parquet"
        output.parent.mkdir(parents=True)
        table = pa.table(
            {
                "video": pa.array(video_values),
                "task_index": pa.array(task_values),
            }
        )
        pq.write_table(table, output)

    def test_matching_outputs_pass(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_episode(root / "single", [[1.0, 2.0], [3.0, 4.0]])
            self.write_episode(
                root / "multi", [[1.0 + 1e-7, 2.0], [3.0, 4.0]]
            )
            compare_datasets(root / "single", root / "multi", rtol=1e-5, atol=1e-6)

    def test_different_outputs_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_episode(root / "single", [[1.0, 2.0], [3.0, 4.0]])
            self.write_episode(root / "multi", [[1.0, 2.0], [3.0, 8.0]])
            with self.assertRaises(AssertionError):
                compare_datasets(
                    root / "single", root / "multi", rtol=1e-5, atol=1e-6
                )


if __name__ == "__main__":
    unittest.main()
