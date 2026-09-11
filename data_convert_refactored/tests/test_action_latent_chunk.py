import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import torch

from data_convert_refactored.encoding.vae_encoding import encode_episode_cpu_friendly


class _ZeroVAE:
    """Minimal VAE stand-in; isolates chunk construction and latent injection."""

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            video.shape[0], 16, 9, 28, 28, dtype=torch.float32, device=video.device
        )


def _transition(action: list[float], proprio: list[float], *, done: bool) -> dict:
    return {
        "state": {
            "video": torch.zeros(1, 3, 33, 224, 224, dtype=torch.uint8),
            "proprio": torch.tensor([proprio], dtype=torch.bfloat16),
        },
        "action": torch.tensor(action, dtype=torch.float32),
        "done": done,
    }


def _install_lightweight_injection_module(monkeypatch) -> None:
    module = ModuleType("cosmos_policy.models.policy_text2world_model")

    def replace_payload(latent, payload, indices):
        batch_indices = torch.arange(latent.shape[0])
        target_elements = latent.shape[1] * latent.shape[3] * latent.shape[4]
        flat = payload.reshape(payload.shape[0], -1)
        repeats = (target_elements + flat.shape[1] - 1) // flat.shape[1]
        filled = flat.repeat(1, repeats)[:, :target_elements].to(latent.dtype)
        latent[batch_indices, :, indices, :, :] = filled.reshape(
            latent.shape[0], latent.shape[1], latent.shape[3], latent.shape[4]
        )
        return latent

    module.replace_latent_with_action_chunk = replace_payload
    module.replace_latent_with_proprio = replace_payload
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_saved_chunk_is_the_exact_payload_injected_at_latent_index_four(monkeypatch):
    # Avoid importing the full Cosmos model (and flash-attn) in this CPU unit test.
    # Production code still imports and calls the official functions.
    _install_lightweight_injection_module(monkeypatch)
    transitions = [
        _transition([0.1, 0.2], [0.3, 0.4], done=False),
        _transition([0.5, 0.6], [0.7, 0.8], done=True),
    ]

    encoded = encode_episode_cpu_friendly(
        transitions,
        encode_batch_size=1,
        device=torch.device("cpu"),
        cosmos_cfg=SimpleNamespace(chunk_size=2, gamma=0.99, tokenizer=None),
        batch_size=1,
        policy=_ZeroVAE(),
    )

    expected_chunks = np.asarray(
        [
            [[0.1, 0.2], [0.5, 0.6]],
            [[0.5, 0.6], [0.5, 0.6]],
        ],
        dtype=np.float32,
    )
    for index, transition in enumerate(encoded):
        saved = transition["action.latent_chunk"].numpy()
        np.testing.assert_array_equal(saved, expected_chunks[index])
        np.testing.assert_array_equal(saved[0], transition["action"].numpy())

        expected_latent_plane = np.resize(saved.reshape(-1), 16 * 28 * 28)
        actual_latent_plane = transition["state"]["video"][0, :, 4].numpy().reshape(-1)
        np.testing.assert_array_equal(actual_latent_plane, expected_latent_plane)
