"""Compatibility wrapper for refactored episode outcome labeling."""

from data_convert_refactored.conditioning.episode_labeling import (
    EpisodeOutcome,
    reward_done_for_step,
)

__all__ = ["EpisodeOutcome", "reward_done_for_step"]
