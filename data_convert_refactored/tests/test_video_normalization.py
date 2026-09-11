import unittest

import torch

from data_convert_refactored.encoding.vae_encoding import normalize_video_batch_cpu


class TestVideoNormalization(unittest.TestCase):
    def test_microbatch_matches_original_formula(self):
        generator = torch.Generator().manual_seed(7)
        video = torch.randint(
            0,
            256,
            (3, 3, 5, 8, 8),
            dtype=torch.uint8,
            generator=generator,
        )

        expected = video.float() / 127.5 - 1.0
        actual = normalize_video_batch_cpu(video)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(video.dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
