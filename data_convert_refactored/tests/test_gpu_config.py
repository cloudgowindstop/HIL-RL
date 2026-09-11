import os
import unittest
from unittest.mock import patch

from data_convert_refactored.encoding.multi_gpu_vae import (
    parse_encode_device_ids,
    resolve_encode_settings,
    validate_encode_settings,
)


class GpuConfigTest(unittest.TestCase):
    def test_parse_device_ids(self):
        self.assertEqual(parse_encode_device_ids("0, 2,7"), (0, 2, 7))
        self.assertIsNone(parse_encode_device_ids(""))
        self.assertIsNone(parse_encode_device_ids(None))

    def test_invalid_device_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_encode_device_ids("0,-1")
        with self.assertRaisesRegex(ValueError, "duplicates"):
            parse_encode_device_ids("1,1")
        with self.assertRaisesRegex(ValueError, "comma-separated integers"):
            parse_encode_device_ids("0,cuda:1")

    def test_world_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, ">= 1"):
            validate_encode_settings(0, None)

    @patch.dict(os.environ, {"ENCODE_WORLD_SIZE": "eight"}, clear=False)
    def test_environment_world_size_must_be_integer(self):
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            resolve_encode_settings(None, None)

    @patch.dict(
        os.environ,
        {"ENCODE_WORLD_SIZE": "4", "ENCODE_CUDA_DEVICES": "1,3"},
        clear=False,
    )
    def test_environment_fallback(self):
        self.assertEqual(resolve_encode_settings(None, None), (4, (1, 3)))

    @patch.dict(
        os.environ,
        {"ENCODE_WORLD_SIZE": "8", "ENCODE_CUDA_DEVICES": "0,1"},
        clear=False,
    )
    def test_cli_overrides_environment(self):
        self.assertEqual(resolve_encode_settings(2, "2,3"), (2, (2, 3)))


if __name__ == "__main__":
    unittest.main()
