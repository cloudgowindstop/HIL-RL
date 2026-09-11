import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from data_convert_refactored.preparation.actions import (
    AUXILIARY_ACTION_FIELDS,
    AUXILIARY_OBSERVATION_FIELDS,
    PuppetNextFrameEncoder,
    encode_transform,
    matrix_to_rotation_6d,
)
from data_convert_refactored.config import ActionEncoding, ActionScale
from data_convert_refactored.preparation.episode import ArmTrajectory, Episode


class ActionTests(unittest.TestCase):
    def test_rotation_6d_uses_first_two_columns(self):
        matrix = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
        encoded = matrix_to_rotation_6d(matrix)
        np.testing.assert_allclose(encoded[:3], matrix[:, 0], atol=1e-7)
        np.testing.assert_allclose(encoded[3:], matrix[:, 1], atol=1e-7)

    def test_metric_transform_is_scaled_once(self):
        delta = np.eye(4)
        delta[:3, 3] = [0.01, -0.02, 0.0]
        action = encode_transform(delta, 0.4, ActionEncoding.ROTATION_6D, ActionScale())
        np.testing.assert_allclose(action[:3], [0.5, -1.0, 0.0])
        self.assertAlmostEqual(float(action[-1]), 0.4)

    def test_puppet_next_frame_repeats_last_valid_action(self):
        poses = np.array(
            [
                [0.00, 0, 0, 0, 0, 0, 1],
                [0.01, 0, 0, 0, 0, 0, 1],
                [0.03, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        arm = ArmTrajectory(poses, np.array([0.0, 0.5, 1.0]))
        episode = Episode(__import__("pathlib").Path("trajectory.hdf5"), 3, ("a", "b"), arm)
        actions = PuppetNextFrameEncoder(ActionEncoding.ROTATION_6D, ActionScale()).encode_episode(
            episode
        )
        self.assertFalse(np.array_equal(actions[0], actions[1]))
        np.testing.assert_array_equal(actions[-1], actions[-2])

    def test_single_frame_puppet_episode_uses_self_delta(self):
        arm = ArmTrajectory(
            np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float64),
            np.array([0.25]),
        )
        episode = Episode(__import__("pathlib").Path("trajectory.hdf5"), 1, ("a", "b"), arm)
        action = PuppetNextFrameEncoder(ActionEncoding.ROTATION_6D, ActionScale()).encode_episode(
            episode
        )[0]
        np.testing.assert_allclose(action[:3], 0.0)
        self.assertAlmostEqual(float(action[-1]), 0.25)

    def test_rotation_6d_conversion_separates_observations_and_actions(self):
        poses = np.zeros((3, 7), dtype=np.float64)
        poses[:, 6] = 1.0
        poses[1, :3] = [0.01, 0.0, 0.0]
        poses[1, 3:7] = Rotation.from_euler("z", 0.2).as_quat()
        poses[2, :3] = [0.02, 0.0, 0.0]
        poses[2, 3:7] = Rotation.from_euler("z", 0.4).as_quat()
        joints = np.array([[0.0, 1.0], [0.1, 1.2], [0.3, 1.5]])
        arm = ArmTrajectory(
            poses,
            np.array([0.0, 0.5, 1.0]),
            puppet_joints=joints,
        )
        episode = Episode(__import__("pathlib").Path("trajectory.hdf5"), 3, ("a", "b"), arm)

        representations = PuppetNextFrameEncoder(
            ActionEncoding.ROTATION_6D, ActionScale()
        ).encode_episode_all(episode)
        auxiliary = representations.auxiliary_fields

        self.assertEqual(
            set(auxiliary), set(AUXILIARY_ACTION_FIELDS + AUXILIARY_OBSERVATION_FIELDS)
        )
        np.testing.assert_allclose(auxiliary["observation.ee_pose_xyzw"][0], poses[0])
        np.testing.assert_allclose(auxiliary["observation.joint_position"][0], joints[0])
        np.testing.assert_allclose(auxiliary["observation.gripper"][0], [0.0])
        np.testing.assert_allclose(auxiliary["action.target_ee_pose_xyzw"][0], poses[1])
        np.testing.assert_allclose(auxiliary["action.target_joint_position"][0], joints[1])
        np.testing.assert_allclose(auxiliary["action.target_gripper"][0], [0.5])
        np.testing.assert_allclose(
            auxiliary["action.delta_joint_position"][0], [0.1, 0.2], atol=1e-7
        )
        np.testing.assert_allclose(
            auxiliary["action.delta_ee_xyz_euler"][0],
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.2],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            auxiliary["action.target_rotation_6d"][0],
            matrix_to_rotation_6d(Rotation.from_euler("z", 0.2).as_matrix()),
        )
        np.testing.assert_allclose(
            auxiliary["observation.rotation_6d"][0],
            matrix_to_rotation_6d(np.eye(3)),
        )
        np.testing.assert_allclose(
            auxiliary["action.delta_rotation_6d"][0],
            matrix_to_rotation_6d(Rotation.from_euler("z", 0.2).as_matrix()),
            atol=1e-7,
        )
        for name in AUXILIARY_ACTION_FIELDS:
            np.testing.assert_array_equal(auxiliary[name][-1], auxiliary[name][-2])
        np.testing.assert_allclose(auxiliary["observation.ee_pose_xyzw"][-1], poses[-1])
        np.testing.assert_allclose(auxiliary["observation.joint_position"][-1], joints[-1])


if __name__ == "__main__":
    unittest.main()
