import unittest

from data_convert_refactored.conditioning.episode_labeling import (
    EpisodeOutcome,
    reward_done_for_step,
)


class EpisodeLabelingTest(unittest.TestCase):
    def test_success_has_five_positive_terminal_frames(self):
        labels = [reward_done_for_step(i, 10, EpisodeOutcome.SUCCESS) for i in range(10)]
        self.assertEqual(labels[:5], [(-0.05, False)] * 5)
        self.assertEqual(labels[5:], [(10.0, True)] * 5)

    def test_failure_is_negative_and_never_done(self):
        labels = [reward_done_for_step(i, 10, EpisodeOutcome.FAILURE) for i in range(10)]
        self.assertEqual(labels, [(-0.05, False)] * 10)

    def test_short_success_marks_every_available_frame(self):
        labels = [reward_done_for_step(i, 3, EpisodeOutcome.SUCCESS) for i in range(3)]
        self.assertEqual(labels, [(10.0, True)] * 3)


if __name__ == "__main__":
    unittest.main()
