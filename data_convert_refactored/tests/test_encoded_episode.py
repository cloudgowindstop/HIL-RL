import unittest
from pathlib import Path

import numpy as np

from data_convert_refactored.config import ActionEncoding, ActionSource
from data_convert_refactored.preparation.actions import EncodedEpisode
from data_convert_refactored.preparation.episode import ArmTrajectory, Episode


def make_episode(length: int = 3) -> Episode:
    pose = np.zeros((length, 7), dtype=np.float32)
    pose[:, 6] = 1.0
    arm = ArmTrajectory(
        puppet_pose_xyzw=pose,
        puppet_gripper=np.zeros(length, dtype=np.float32),
    )
    return Episode(
        path=Path("trajectory.hdf5"),
        length=length,
        camera_names=("primary", "wrist"),
        left=arm,
    )


class EncodedEpisodeTests(unittest.TestCase):
    def test_accepts_finite_precomputed_actions(self):
        encoded = EncodedEpisode(
            episode=make_episode(),
            actions=np.zeros((3, 10), dtype=np.float64),
            action_source=ActionSource.PUPPET_NEXT_FRAME,
            action_encoding=ActionEncoding.ROTATION_6D,
            last_action_is_padding=True,
        )
        self.assertEqual(encoded.action_dim, 10)
        self.assertEqual(encoded.actions.dtype, np.float32)

    def test_rejects_action_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "action length"):
            EncodedEpisode(
                episode=make_episode(),
                actions=np.zeros((2, 10), dtype=np.float32),
                action_source=ActionSource.PUPPET_NEXT_FRAME,
                action_encoding=ActionEncoding.ROTATION_6D,
                last_action_is_padding=True,
            )


if __name__ == "__main__":
    unittest.main()
