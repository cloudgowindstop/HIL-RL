from typing import Dict, Iterable, Optional, Tuple
import os
import numpy as np
from PIL import Image

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
# done by zxy for conrft
def resize_image(image, target_size):
    """Resize image to target size using bilinear interpolation."""
    if image.shape[-3:-1] != target_size:
        # Use jax.image.resize for bilinear interpolation
        new_shape = list(image.shape)
        new_shape[-3:-1] = target_size
        return jax.image.resize(image, new_shape, method="bilinear")
    return image

class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
    """

    encoder: nn.Module
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)
    dtype: any = jnp.float32
    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train=False,
        stop_gradient=False,
        is_encoded=False,
    ) -> jnp.ndarray:
        observations = {k: v.astype(self.dtype) for k, v in observations.items()}
        # encode images with encoder
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key].astype(self.dtype)
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:
                        T = image.shape[0]
                        if T > 1:
                            image = image[-1:] # for stacked images, only use the last frame
                        image = rearrange(image, "T H W C -> H W (T C)")
                    if len(image.shape) == 5:
                        T = image.shape[1]
                        if T > 1:
                            image = image[:, -1:] # for stacked images, only use the last frame
                        image = rearrange(image, "B T H W C -> B H W (T C)")

            image = self.encoder[image_key](image, train=train, encode=not is_encoded)

            if stop_gradient:
                image = jax.lax.stop_gradient(image)

            encoded.append(image)

        encoded = jnp.concatenate(encoded, axis=-1)

        if self.use_proprio:
            # project state to embeddings as well
            state = observations["state"].astype(self.dtype)
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    encoded = encoded.reshape(-1)
                if len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")
            state = nn.Dense(
                self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform(), dtype=self.dtype
            )(state)
            state = nn.LayerNorm(dtype=self.dtype)(state)
            state = nn.tanh(state)
            encoded = jnp.concatenate([encoded, state], axis=-1)

        return encoded
    
