"""Primitive specification shared by synthetic input and independent oracle.

This module contains only literal test inputs. It must not derive poses,
actions, proprio, temporal chunks, returns, or latent values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticScenario:
    """Declarative inputs for one deterministic single-arm conversion test."""

    length: int = 20
    source_fps: int = 30
    translation_step_m: float = 0.01
    rotation_step_rad: float = 0.03
    translation_scale_m: float = 0.02
    rotation_scale_rad: float = 0.06
    gripper_scale: float = 1.0
    action_dimension: int = 7
    proprio_dimension: int = 8
    chunk_size: int = 16
    gamma: float = 0.99
    action_min: tuple[float, ...] = (-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 0.0)
    action_max: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0)
    proprio_min: tuple[float, ...] = (-1.0, -2.0, -3.0, -1.0, -1.5, -2.0, -0.5, 0.0)
    proprio_max: tuple[float, ...] = (1.0, 2.0, 3.0, 1.0, 1.5, 2.0, 1.5, 1.0)


SCENARIO = SyntheticScenario()

