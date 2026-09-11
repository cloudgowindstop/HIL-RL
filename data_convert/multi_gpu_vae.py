"""Compatibility wrapper for the refactored multi-GPU VAE implementation."""

from data_convert_refactored.encoding.multi_gpu_vae import (  # noqa: F401
    VAEReplicaPool,
    get_or_create_vae_replica_pool,
    instantiate_tokenizer_replica,
    tokenizer_device,
    validate_tokenizer_replica,
)
