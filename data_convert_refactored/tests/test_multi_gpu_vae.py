import os
from types import SimpleNamespace
import unittest

import torch

from data_convert_refactored.encoding.multi_gpu_vae import (
    get_or_create_vae_replica_pool,
    validate_tokenizer_replica,
)


class FakeWanBackend:
    def __init__(self, device):
        self.model = torch.nn.Linear(1, 1, bias=False).to(device)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
        for name in (
            "mean",
            "std",
            "img_mean",
            "img_std",
            "video_mean",
            "video_std",
        ):
            setattr(self, name, torch.ones(1, device=device))


class FakeWanInterface:
    """Production-shaped wrapper: intentionally has no parameters/to/eval methods."""

    def __init__(self, device):
        self.model = FakeWanBackend(device)

    def encode(self, batch):
        scale = self.model.model.weight.reshape(1, 1, 1, 1, 1)
        return batch * scale


class FakePolicy:
    def __init__(self, primary_device="cuda:0"):
        self.tokenizer = FakeWanInterface(primary_device)
        self.sigma_data = 1.0

    @staticmethod
    def _resolve_encode_devices(world_size, encode_device_ids, num_steps):
        ids = encode_device_ids if encode_device_ids is not None else list(range(world_size))
        return [f"cuda:{device_id}" for device_id in ids[:num_steps]]

    @staticmethod
    def _shard_lengths(num_items, world_size):
        base, remainder = divmod(num_items, world_size)
        return [base + (1 if index < remainder else 0) for index in range(world_size)]


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices are required")
class VAEReplicaPoolTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.set_device(0)
        self.factory_calls = []

    def factory(self, _config, device):
        self.factory_calls.append(str(device))
        return FakeWanInterface(device)

    def test_complete_wrappers_are_cached_and_preserve_order(self):
        policy = FakePolicy()
        pool = get_or_create_vae_replica_pool(
            policy=policy,
            tokenizer_config={"name": "fake"},
            encode_world_size=2,
            encode_device_ids=(0, 1),
            encode_batch_size=4,
            tokenizer_factory=self.factory,
        )
        self.assertEqual(self.factory_calls, ["cuda:1"])
        self.assertIs(pool.replicas["cuda:0"], policy.tokenizer)
        validate_tokenizer_replica(pool.replicas["cuda:1"], torch.device("cuda:1"))

        batch = torch.arange(7, dtype=torch.float32).reshape(7, 1, 1, 1, 1)
        actual = pool.encode(batch, sigma_data=1.0)
        self.assertEqual(actual.device.type, "cpu")
        torch.testing.assert_close(actual, batch)

        cached = get_or_create_vae_replica_pool(
            policy=policy,
            tokenizer_config={"name": "fake"},
            encode_world_size=2,
            encode_device_ids=(0, 1),
            encode_batch_size=4,
            tokenizer_factory=self.factory,
        )
        self.assertIs(cached, pool)
        self.assertEqual(self.factory_calls, ["cuda:1"])

    def test_wrong_replica_device_is_rejected(self):
        policy = FakePolicy()

        def wrong_factory(_config, _device):
            return FakeWanInterface("cuda:0")

        with self.assertRaisesRegex(RuntimeError, "expected on cuda:1"):
            get_or_create_vae_replica_pool(
                policy=policy,
                tokenizer_config={"name": "fake"},
                encode_world_size=2,
                encode_device_ids=(0, 1),
                encode_batch_size=4,
                tokenizer_factory=wrong_factory,
            )

@unittest.skipUnless(
    os.environ.get("RUN_REAL_COSMOS_VAE_TEST") == "1",
    "set RUN_REAL_COSMOS_VAE_TEST=1 to load real VAE wrappers",
)
class RealCosmosVAEReplicaTest(unittest.TestCase):
    def test_real_multi_gpu_wrappers_match_single_gpu(self):
        world_size = int(os.environ.get("REAL_COSMOS_VAE_WORLD_SIZE", "2"))
        if world_size < 2:
            self.fail("REAL_COSMOS_VAE_WORLD_SIZE must be >= 2")
        if torch.cuda.device_count() < world_size:
            self.skipTest(f"{world_size} CUDA devices are required")

        from data_convert_refactored.encoding.policy_loader import init_cosmos_policy

        config = SimpleNamespace(
            input_dir="",
            output_dir="",
            task_description="VAE replica smoke test",
            cosmos_config_path="train_config_cosmos.json",
            encode_batch_size=world_size,
            encode_world_size=world_size,
            encode_device_ids=tuple(range(world_size)),
            resume=False,
        )
        policy, _ = init_cosmos_policy(config)
        pool = getattr(policy, "_conversion_vae_replica_pool")

        batch = torch.zeros((world_size, 3, 33, 224, 224), dtype=torch.float32)
        # Compare identical per-device batch shapes. A single batch of two uses
        # different bfloat16 kernels than two one-item shards and can differ by
        # normal low-precision rounding even with identical weights.
        expected_parts = []
        with torch.inference_mode(), torch.cuda.device(0):
            for index in range(batch.shape[0]):
                expected_parts.append(
                    policy.encode(batch[index : index + 1].to("cuda:0"))
                    .contiguous()
                    .float()
                    .cpu()
                )
        expected = torch.cat(expected_parts, dim=0)
        actual = pool.encode(batch, sigma_data=policy.sigma_data)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
