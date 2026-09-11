import unittest
from pathlib import Path

import numpy as np

from data_convert_refactored.preparation.actions import EncodedEpisode, normalize_encoded_episodes
from data_convert_refactored.config import ActionEncoding, ActionSource
from data_convert_refactored.preparation.dataset_stats import compute_action_stats
from data_convert_refactored.preparation.episode import ArmTrajectory, Episode


def encoded(actions: np.ndarray) -> EncodedEpisode:
    length = len(actions)
    pose = np.zeros((length, 7), dtype=np.float32)
    pose[:, 6] = 1.0
    episode = Episode(
        path=Path("trajectory.hdf5"),
        length=length,
        camera_names=("primary", "wrist"),
        left=ArmTrajectory(pose, np.zeros(length, dtype=np.float32)),
    )
    return EncodedEpisode(
        episode=episode,
        actions=actions,
        action_source=ActionSource.PUPPET_NEXT_FRAME,
        action_encoding=ActionEncoding.ROTATION_6D,
        last_action_is_padding=True,
    )


def reference_rescale(action, stats, non_negative_only=False, scale_multiplier=1.0):
    assert not non_negative_only
    return scale_multiplier * (
        2 * (action - stats["actions_min"])
        / (stats["actions_max"] - stats["actions_min"])
        - 1
    )


class ActionNormalizationTests(unittest.TestCase):
    def test_full_dataset_statistics_and_collect_formula(self):
        episodes = [
            encoded(np.array([[0.0, 2.0], [1.0, 4.0]], dtype=np.float32)),
            encoded(np.array([[2.0, 6.0]], dtype=np.float32)),
        ]
        stats = compute_action_stats(episodes)
        normalized = normalize_encoded_episodes(episodes, stats, reference_rescale)
        np.testing.assert_allclose(stats["actions_min"], [0.0, 2.0])
        np.testing.assert_allclose(stats["actions_max"], [2.0, 6.0])
        np.testing.assert_allclose(normalized[0].actions, [[-1.0, -1.0], [0.0, 0.0]])
        np.testing.assert_allclose(normalized[1].actions, [[1.0, 1.0]])

    def test_constant_channel_fails_before_division_by_zero(self):
        with self.assertRaisesRegex(ValueError, "constant channels"):
            compute_action_stats(
                [encoded(np.array([[0.0, 1.0], [2.0, 1.0]], dtype=np.float32))]
            )


if __name__ == "__main__":
    unittest.main()
