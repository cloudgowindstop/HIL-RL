"""Independent numerical oracle for the synthetic single-arm episode.

This module intentionally does not import action, normalization, or temporal
helpers from the production converter. A regression in production code must
therefore disagree with this oracle instead of being repeated on both sides of
the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scenario_spec import SCENARIO, SyntheticScenario


@dataclass(frozen=True)
class SyntheticExpectedValues:
    """Complete low-dimensional reference values for one synthetic episode."""

    poses_xyzw: np.ndarray
    gripper: np.ndarray
    first_stage_actions: np.ndarray
    rotation_6d_actions: np.ndarray
    normalized_actions: np.ndarray
    raw_proprio: np.ndarray
    normalized_proprio: np.ndarray
    action_chunks: np.ndarray
    future_proprio: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    returns: np.ndarray
    future_returns: np.ndarray


def poses_xyzw(scenario: SyntheticScenario = SCENARIO) -> np.ndarray:
    """Derive every pose with closed-form trigonometric sums.

    This does not use the generator's recursive homogeneous transforms.
    """
    poses = np.zeros((scenario.length, 7), dtype=np.float64)
    half_step = scenario.rotation_step_rad / 2.0
    denominator = np.sin(half_step)
    for timestep in range(scenario.length):
        theta = timestep * scenario.rotation_step_rad
        if timestep:
            common = np.sin(timestep * half_step) / denominator
            poses[timestep, 0] = (
                scenario.translation_step_m
                * common
                * np.cos((timestep - 1) * half_step)
            )
            poses[timestep, 1] = (
                scenario.translation_step_m
                * common
                * np.sin((timestep - 1) * half_step)
            )
        poses[timestep, 5] = np.sin(theta / 2.0)
        poses[timestep, 6] = np.cos(theta / 2.0)
    return poses.astype(np.float32)


def gripper_positions(scenario: SyntheticScenario = SCENARIO) -> np.ndarray:
    """Use a distinct monotonic value at every timestep."""
    return np.linspace(0.0, 1.0, scenario.length, dtype=np.float32)


def first_stage_actions(scenario: SyntheticScenario = SCENARIO) -> np.ndarray:
    """Compute every Euler action without calling production transform code."""
    gripper = gripper_positions(scenario)
    actions = np.zeros(
        (scenario.length, scenario.action_dimension), dtype=np.float32
    )
    actions[:-1, 0] = scenario.translation_step_m / scenario.translation_scale_m
    actions[:-1, 5] = scenario.rotation_step_rad / scenario.rotation_scale_rad
    actions[:-1, 6] = gripper[1:] / scenario.gripper_scale
    actions[-1] = actions[-2]
    return actions


def rotation_6d_actions(scenario: SyntheticScenario = SCENARIO) -> np.ndarray:
    """Independently encode the known local Z rotation as standard rotation-6D."""
    actions = np.zeros((scenario.length, 10), dtype=np.float32)
    cosine = np.cos(scenario.rotation_step_rad)
    sine = np.sin(scenario.rotation_step_rad)
    rotation_6d = np.array(
        [cosine, sine, 0.0, -sine, cosine, 0.0], dtype=np.float32
    )
    actions[:-1, 0] = scenario.translation_step_m / scenario.translation_scale_m
    actions[:-1, 3:9] = rotation_6d
    actions[:-1, 9] = gripper_positions(scenario)[1:] / scenario.gripper_scale
    actions[-1] = actions[-2]
    return actions


def scenario_stats(scenario: SyntheticScenario = SCENARIO) -> dict[str, np.ndarray]:
    """Materialize non-identity per-channel stats from literal scenario inputs."""
    return {
        "actions_min": np.asarray(scenario.action_min, dtype=np.float32),
        "actions_max": np.asarray(scenario.action_max, dtype=np.float32),
        "proprio_min": np.asarray(scenario.proprio_min, dtype=np.float32),
        "proprio_max": np.asarray(scenario.proprio_max, dtype=np.float32),
    }


def minmax_normalize(
    values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> np.ndarray:
    """Independently apply the official [-1, 1] min/max formula."""
    return (2.0 * ((values - minimum) / (maximum - minimum)) - 1.0).astype(
        np.float32
    )


def raw_proprio(scenario: SyntheticScenario = SCENARIO) -> np.ndarray:
    """Build xyz + xyzw quaternion + absolute gripper in production order."""
    return np.concatenate(
        (poses_xyzw(scenario), gripper_positions(scenario)[:, None]), axis=1
    ).astype(np.float32)


def action_chunks(actions: np.ndarray, chunk_size: int = SCENARIO.chunk_size) -> np.ndarray:
    """Build every temporal chunk and repeat the final action at the boundary."""
    length, action_dimension = actions.shape
    chunks = np.empty((length, chunk_size, action_dimension), dtype=np.float32)
    for timestep in range(length):
        for offset in range(chunk_size):
            chunks[timestep, offset] = actions[min(timestep + offset, length - 1)]
    return chunks


def future_values(values: np.ndarray, chunk_size: int = SCENARIO.chunk_size) -> np.ndarray:
    """Select min(t + chunk_size, T - 1) for every timestep."""
    length = len(values)
    return np.stack(
        [values[min(timestep + chunk_size, length - 1)] for timestep in range(length)]
    )


def reward_done(length: int = SCENARIO.length) -> tuple[np.ndarray, np.ndarray]:
    """Apply the explicit successful-episode rule independently."""
    rewards = np.full(length, -0.05, dtype=np.float32)
    dones = np.zeros(length, dtype=bool)
    rewards[-5:] = 10.0
    dones[-5:] = True
    return rewards, dones


def monte_carlo_returns(
    length: int = SCENARIO.length, gamma: float = SCENARIO.gamma
) -> np.ndarray:
    """Closed-form return for terminal reward 1, rescaled from [0, 1] to [-1, 1]."""
    exponent = np.arange(length - 1, -1, -1, dtype=np.float64)
    return (2.0 * np.power(gamma, exponent) - 1.0).astype(np.float32)


def build_expected_values(
    scenario: SyntheticScenario = SCENARIO,
) -> SyntheticExpectedValues:
    """Construct all low-dimensional values checked before and after VAE encoding."""
    stats = scenario_stats(scenario)
    actions = first_stage_actions(scenario)
    proprio = raw_proprio(scenario)
    normalized_actions = minmax_normalize(
        actions, stats["actions_min"], stats["actions_max"]
    )
    normalized_proprio = minmax_normalize(
        proprio, stats["proprio_min"], stats["proprio_max"]
    )
    returns = monte_carlo_returns(scenario.length, scenario.gamma)
    rewards, dones = reward_done(scenario.length)
    return SyntheticExpectedValues(
        poses_xyzw=poses_xyzw(scenario),
        gripper=gripper_positions(scenario),
        first_stage_actions=actions,
        rotation_6d_actions=rotation_6d_actions(scenario),
        normalized_actions=normalized_actions,
        raw_proprio=proprio,
        normalized_proprio=normalized_proprio,
        action_chunks=action_chunks(normalized_actions, scenario.chunk_size),
        future_proprio=future_values(normalized_proprio, scenario.chunk_size),
        rewards=rewards,
        dones=dones,
        returns=returns,
        future_returns=future_values(returns, scenario.chunk_size),
    )


def inject_repeated_condition(
    latent: np.ndarray, values: np.ndarray, latent_index: int
) -> None:
    """Fill one latent time plane by repeating flattened conditions.

    This NumPy implementation is independent of Cosmos
    ``replace_latent_with_action_chunk`` and ``replace_latent_with_proprio``.
    """
    batch_size, channels, _, height, width = latent.shape
    flat_values = values.reshape(batch_size, -1)
    plane_elements = channels * height * width
    if flat_values.shape[1] > plane_elements:
        raise ValueError("condition does not fit in one latent time plane")
    repeats = (plane_elements + flat_values.shape[1] - 1) // flat_values.shape[1]
    repeated = np.tile(flat_values, (1, repeats))[:, :plane_elements]
    latent[:, :, latent_index, :, :] = repeated.reshape(
        batch_size, channels, height, width
    )


def inject_all_conditions(
    base_latent: np.ndarray, expected: SyntheticExpectedValues
) -> np.ndarray:
    """Build complete expected latent using independently computed conditions."""
    output = np.asarray(base_latent, dtype=np.float32).copy()
    # Production transitions store proprio as bfloat16 before VAE injection.
    import torch

    proprio = (
        torch.from_numpy(expected.normalized_proprio)
        .to(torch.bfloat16)
        .float()
        .numpy()
    )
    future_proprio = (
        torch.from_numpy(expected.future_proprio)
        .to(torch.bfloat16)
        .float()
        .numpy()
    )
    inject_repeated_condition(output, proprio, 1)
    inject_repeated_condition(output, expected.action_chunks, 4)
    inject_repeated_condition(output, future_proprio, 5)
    output[:, :, 8, :, :] = expected.future_returns[:, None, None, None]
    return output
