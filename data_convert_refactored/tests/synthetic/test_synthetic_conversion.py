"""CPU regression tests for every predictable pre-VAE conversion value."""

from __future__ import annotations

import tempfile
import unittest
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cosmos_policy.datasets.dataset_common import (
    compute_monte_carlo_returns,
    get_action_chunk_with_padding,
)

from data_convert_refactored.conditioning.transition_builder import (
    build_transition_list_for_episode,
)
from data_convert_refactored.config import ActionEncoding, ActionScale
from data_convert_refactored.preparation.actions import PuppetNextFrameEncoder
from data_convert_refactored.preparation.episode import load_episode
from data_convert_refactored.preparation.proprio import raw_proprio_for_episode
from data_convert_refactored.tests.synthetic.hdf5_generator import (
    create_synthetic_fixture,
)
from data_convert_refactored.tests.synthetic import expected_values, hdf5_generator
from data_convert_refactored.tests.synthetic.expected_values import (
    build_expected_values,
    inject_all_conditions,
    scenario_stats,
)


class SyntheticConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.hdf5_path, _ = create_synthetic_fixture(self.root)
        self.expected = build_expected_values()
        self.episode = load_episode(self.hdf5_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loader_and_every_low_dimensional_value(self) -> None:
        np.testing.assert_allclose(
            self.episode.left.puppet_pose_xyzw,
            self.expected.poses_xyzw,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            raw_proprio_for_episode(self.episode),
            self.expected.raw_proprio,
            rtol=0.0,
            atol=1e-7,
        )
        encoder = PuppetNextFrameEncoder(
            ActionEncoding.LEGACY_EULER,
            ActionScale(0.02, 0.06, 1.0),
        )
        representations = encoder.encode_episode_all(self.episode)
        actions = representations.euler_control
        np.testing.assert_allclose(
            actions,
            self.expected.first_stage_actions,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(actions[-1], actions[-2], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            representations.rotation_6d_control,
            self.expected.rotation_6d_actions,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_generator_and_oracle_have_no_implementation_dependency(self) -> None:
        generator_source = inspect.getsource(hdf5_generator)
        oracle_source = inspect.getsource(expected_values)
        self.assertNotIn("expected_values", generator_source)
        self.assertNotIn("hdf5_generator", oracle_source)
        self.assertNotIn("h5py", oracle_source)
        self.assertNotIn("load_episode", oracle_source)

    def test_stats_are_non_identity_and_channel_specific(self) -> None:
        stats = scenario_stats()
        self.assertGreater(len(np.unique(stats["actions_min"])), 2)
        self.assertGreater(len(np.unique(stats["actions_max"])), 2)
        self.assertFalse(
            np.array_equal(
                self.expected.first_stage_actions,
                self.expected.normalized_actions,
            )
        )

    def test_transition_values_and_camera_markers(self) -> None:
        config = SimpleNamespace(
            action_encoding="legacy_euler",
            action_source="puppet_next_frame",
            episode_outcome="success",
            image_size=224,
            wrist_crop_mode="center_width",
            wrist_crop_fraction=0.75,
            wrist_left_center_offset_x=64,
            wrist_right_center_offset_x=64,
            head_crop_top_pixels=100,
            use_jpeg_compression=False,
            trained_with_image_aug=False,
        )
        representations = PuppetNextFrameEncoder(
            ActionEncoding.ROTATION_6D,
            ActionScale(0.02, 0.06, 1.0),
        ).encode_episode_all(self.episode)
        transitions, info = build_transition_list_for_episode(
            self.episode,
            "single_absolute",
            None,
            config,
            normalized_actions=self.expected.normalized_actions,
            normalized_proprio=self.expected.normalized_proprio,
            euler_actions=self.expected.first_stage_actions,
            rotation_6d_actions=self.expected.rotation_6d_actions,
            auxiliary_fields=representations.auxiliary_fields,
        )
        self.assertEqual(info["action_dim"], 7)
        self.assertEqual(info["proprio_dim"], 8)
        self.assertEqual(info["euler_action_dim"], 7)
        self.assertEqual(info["rotation_6d_action_dim"], 10)
        self.assertEqual(
            set(info["auxiliary_fields"]),
            {
                "action.target_rotation_6d",
                "action.delta_rotation_6d",
                "action.target_ee_pose_xyzw",
                "action.target_joint_position",
                "action.delta_ee_xyz_euler",
                "action.delta_joint_position",
                "action.target_gripper",
                "observation.rotation_6d",
                "observation.ee_pose_xyzw",
                "observation.joint_position",
                "observation.gripper",
            },
        )
        actual_rewards = np.asarray([item["reward"] for item in transitions])
        actual_dones = np.asarray([item["done"] for item in transitions])
        np.testing.assert_allclose(actual_rewards, self.expected.rewards, atol=1e-7)
        np.testing.assert_array_equal(actual_dones, self.expected.dones)

        for timestep, transition in enumerate(transitions):
            np.testing.assert_allclose(
                transition["action.euler_control"].numpy(),
                self.expected.first_stage_actions[timestep],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                transition["action.rotation_6d_control"].numpy(),
                self.expected.rotation_6d_actions[timestep],
                atol=1e-6,
            )
            for name, values in representations.auxiliary_fields.items():
                np.testing.assert_allclose(
                    transition[name].numpy(), values[timestep], atol=1e-6
                )
            np.testing.assert_array_equal(
                transition["source.puppet.pose_xyzw"].numpy(),
                self.episode.left.puppet_pose_xyzw[timestep],
            )
            np.testing.assert_array_equal(
                transition["source.puppet.gripper"].numpy(),
                self.episode.left.puppet_gripper[timestep : timestep + 1],
            )
            video = transition["state"]["video"].numpy()
            self.assertEqual(video.shape, (1, 3, 33, 224, 224))
            wrist_red = 20 + timestep * 7
            primary_red = 80 + timestep * 7
            np.testing.assert_array_equal(video[0, 0, 5:9], wrist_red)
            np.testing.assert_array_equal(video[0, 0, 9:13], primary_red)

    def test_independent_temporal_oracle_matches_official_helpers(self) -> None:
        official_chunks = np.stack(
            [
                get_action_chunk_with_padding(
                    self.expected.normalized_actions,
                    timestep,
                    16,
                    len(self.expected.normalized_actions),
                )
                for timestep in range(len(self.expected.normalized_actions))
            ]
        )
        np.testing.assert_allclose(
            official_chunks, self.expected.action_chunks, rtol=0.0, atol=0.0
        )
        official_returns = compute_monte_carlo_returns(
            len(self.expected.rewards), terminal_reward=1.0, gamma=0.99
        )
        np.testing.assert_allclose(
            official_returns, self.expected.returns, rtol=1e-6, atol=1e-7
        )

    def test_independent_latent_injection_fills_exact_expected_values(self) -> None:
        generator = np.random.default_rng(7)
        base = generator.standard_normal((20, 16, 9, 28, 28)).astype(np.float32)
        independent = inject_all_conditions(base, self.expected)
        np.testing.assert_array_equal(independent[:, :, 0], base[:, :, 0])
        np.testing.assert_array_equal(
            independent[0, :, 4].reshape(-1)[:112],
            self.expected.action_chunks[0].reshape(-1),
        )
        expected_proprio = (
            torch.from_numpy(self.expected.normalized_proprio)
            .to(torch.bfloat16)
            .float()
            .numpy()
        )
        np.testing.assert_array_equal(
            independent[0, :, 1].reshape(-1)[:8], expected_proprio[0]
        )
        np.testing.assert_array_equal(
            independent[:, 0, 8, 0, 0], self.expected.future_returns
        )


if __name__ == "__main__":
    unittest.main()
