"""显式episode结果标签及逐帧reward/done规则。"""

from __future__ import annotations

from enum import Enum


class EpisodeOutcome(str, Enum):
    """由CLI显式指定，不从图像或HDF5内容自动推断。"""

    SUCCESS = "success"
    FAILURE = "failure"


def reward_done_for_step(
    timestep: int,
    episode_length: int,
    outcome: EpisodeOutcome | str,
    *,
    success_tail_frames: int = 5,
    reward_positive: float = 10.0,
    reward_negative: float = -0.05,
) -> tuple[float, bool]:
    """返回当前帧reward和done。

    成功轨迹最后5帧为(10.0, True)，此前为(-0.05, False)；失败轨迹所有帧均为
    (-0.05, False)。短于5帧的成功轨迹全部属于成功尾段。
    """
    outcome = EpisodeOutcome(outcome)
    if episode_length < 1:
        raise ValueError("episode_length must be positive")
    if not 0 <= timestep < episode_length:
        raise ValueError(f"timestep {timestep} is outside episode length {episode_length}")
    if success_tail_frames < 1:
        raise ValueError("success_tail_frames must be positive")

    if outcome is EpisodeOutcome.FAILURE:
        return float(reward_negative), False

    is_success_tail = timestep >= max(0, episode_length - success_tail_frames)
    return (
        float(reward_positive if is_success_tail else reward_negative),
        bool(is_success_tail),
    )
