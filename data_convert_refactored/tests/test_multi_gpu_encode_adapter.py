import unittest
from unittest.mock import Mock, patch

import torch

from data_convert_refactored.encoding.vae_encoding import encode_normalized_video_batch


class FakePolicy:
    def __init__(self):
        self.single_calls = 0
        self.sigma_data = 1.0

    def encode(self, batch):
        self.single_calls += 1
        return batch + 1


class MultiGpuEncodeAdapterTest(unittest.TestCase):
    def setUp(self):
        self.batch = torch.zeros((4, 3, 2, 2, 2), dtype=torch.float32)

    def test_default_path_remains_policy_encode(self):
        policy = FakePolicy()
        actual = encode_normalized_video_batch(
            policy,
            self.batch,
            single_device=torch.device("cpu"),
            encode_batch_size=4,
        )
        self.assertEqual(policy.single_calls, 1)
        torch.testing.assert_close(actual, self.batch + 1)

    @patch("data_convert_refactored.encoding.multi_gpu_vae.get_or_create_vae_replica_pool")
    def test_world_size_routes_to_conversion_replica_pool(self, get_pool):
        policy = FakePolicy()
        pool = Mock()
        pool.encode.return_value = self.batch + 2
        get_pool.return_value = pool

        actual = encode_normalized_video_batch(
            policy,
            self.batch,
            single_device=torch.device("cpu"),
            encode_batch_size=4,
            encode_world_size=2,
            tokenizer_config={"name": "fake"},
        )

        self.assertEqual(policy.single_calls, 0)
        get_pool.assert_called_once_with(
            policy=policy,
            tokenizer_config={"name": "fake"},
            encode_world_size=2,
            encode_device_ids=None,
            encode_batch_size=4,
        )
        pool.encode.assert_called_once_with(self.batch, sigma_data=1.0)
        torch.testing.assert_close(actual, self.batch + 2)

    @patch("data_convert_refactored.encoding.multi_gpu_vae.get_or_create_vae_replica_pool")
    def test_explicit_devices_are_passed_to_replica_pool(self, get_pool):
        policy = FakePolicy()
        pool = Mock()
        pool.encode.return_value = self.batch
        get_pool.return_value = pool

        encode_normalized_video_batch(
            policy,
            self.batch,
            single_device=torch.device("cpu"),
            encode_batch_size=8,
            encode_world_size=1,
            encode_device_ids=(1, 3),
            tokenizer_config={"name": "fake"},
        )

        self.assertEqual(get_pool.call_args.kwargs["encode_device_ids"], (1, 3))
        self.assertEqual(get_pool.call_args.kwargs["encode_batch_size"], 8)

    def test_multi_gpu_requires_tokenizer_config(self):
        with self.assertRaisesRegex(ValueError, "tokenizer_config"):
            encode_normalized_video_batch(
                FakePolicy(),
                self.batch,
                single_device=torch.device("cpu"),
                encode_batch_size=4,
                encode_world_size=2,
            )

    @patch("data_convert_refactored.encoding.multi_gpu_vae.get_or_create_vae_replica_pool")
    def test_latent_length_mismatch_is_rejected(self, get_pool):
        pool = Mock()
        pool.encode.return_value = self.batch[:-1]
        get_pool.return_value = pool
        with self.assertRaisesRegex(RuntimeError, "latent batch length"):
            encode_normalized_video_batch(
                FakePolicy(),
                self.batch,
                single_device=torch.device("cpu"),
                encode_batch_size=4,
                encode_world_size=2,
                tokenizer_config={"name": "fake"},
            )

    def test_input_must_remain_on_cpu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to create a non-CPU tensor")
        with self.assertRaisesRegex(ValueError, "must stay on CPU"):
            encode_normalized_video_batch(
                FakePolicy(),
                self.batch.cuda(),
                single_device=torch.device("cuda"),
                encode_batch_size=4,
            )


if __name__ == "__main__":
    unittest.main()
