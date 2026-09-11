import numpy as np
import torch

from data_convert_refactored.dataset_writer import (
    build_output_features,
    cosmos_encoded_state_to_frame,
)


def test_clean_restore_feature_is_optional_and_compact_float16():
    legacy = build_output_features(20, 16)
    enabled = build_output_features(20, 16, save_clean_restore_latent=True)

    assert legacy["action.latent_chunk"] == {
        "dtype": "float32",
        "shape": (16, 20),
        "names": None,
    }
    assert "clean_restore_latent" not in legacy
    assert enabled["clean_restore_latent"] == {
        "dtype": "float16",
        "shape": (16, 4, 28, 28),
        "names": None,
    }


def test_action_latent_chunk_schema_uses_runtime_chunk_size():
    features = build_output_features(14, 16, action_chunk_size=8)

    assert features["action.latent_chunk"]["shape"] == (8, 14)


def test_state_conversion_preserves_restore_payload_float16():
    state = {
        "video": torch.zeros(1, 16, 9, 28, 28),
        "clean_restore_latent": torch.ones(1, 16, 4, 28, 28, dtype=torch.float16),
    }
    frame = cosmos_encoded_state_to_frame(state)

    assert frame["video"].dtype == np.float32
    assert frame["clean_restore_latent"].dtype == np.float16
    assert frame["clean_restore_latent"].shape == (16, 4, 28, 28)
