"""数据转换顶层流程：准备episode、计算action、校验stats并启动Cosmos后端。"""

from __future__ import annotations

import json

from .preparation.actions import (
    AUXILIARY_ACTION_FIELDS,
    AUXILIARY_OBSERVATION_FIELDS,
    EncodedEpisode,
    make_action_encoder,
)
from .config import ActionSource, ConversionConfig
from .cosmos_backend import CosmosBackend
from .preparation.dataset_stats import (
    prepare_dataset_stats,
    validate_official_stats,
    validate_external_stats,
    validate_or_warn_stats_ranges,
)
from .config import StatsMode
from .preparation.episode import discover_episodes, load_episode
from .preparation.kinematics import DualArmKinematics
from .preflight import PreflightReport, run_preflight
from .shape_trace import require_shape, trace_shape


class ConversionPipeline:
    """组织一次完整转换，但不实现具体的机器人动作或VAE算法。"""

    def __init__(self, config: ConversionConfig):
        self.config = config
        self.kinematics: DualArmKinematics | None = None
        self.resume_state = None

    def prepare(self, preflight_only: bool = False) -> tuple[list, PreflightReport | None]:
        """发现并读取HDF5，同时完成机器人类型、维度和可选FK预检。"""
        self.config.validate()
        paths = discover_episodes(
            self.config.input_dir,
            self.config.max_episodes,
            self.config.episode_manifest,
        )
        if not paths:
            raise ValueError(f"no trajectory.hdf5 found under {self.config.input_dir}")
        if not preflight_only:
            from .resume import prepare_resume

            self.resume_state = prepare_resume(self.config, paths, enabled=self.config.resume)
            paths = paths[self.resume_state.start_episode :]
            if not paths:
                return [], None
        episodes = [load_episode(path) for path in paths]
        if self.config.action_source is ActionSource.MASTER_JOINT_FK_SAME_FRAME:
            self.kinematics = DualArmKinematics(self.config.kinematics_config)
        report = run_preflight(self.config, episodes, self.kinematics)
        return episodes, report

    def create_action_encoder(self):
        """根据CLI语义选择action来源；encoder负责生成未做dataset归一化的(T,D)。"""
        return make_action_encoder(
            self.config.action_source,
            self.config.action_encoding,
            self.config.action_scale,
            self.kinematics,
        )

    def inspect_actions(self, encoded_episode: EncodedEpisode) -> dict:
        """生成便于终端审查的action摘要，不修改action内容。"""
        action = encoded_episode.actions
        return {
            "episode": str(encoded_episode.episode.path),
            "shape": list(action.shape),
            "finite": bool(__import__("numpy").isfinite(action).all()),
            "minimum": action.min(axis=0).tolist(),
            "maximum": action.max(axis=0).tolist(),
            "last_action_is_padding": encoded_episode.last_action_is_padding,
        }

    def encode_actions(self, episodes: list) -> list[EncodedEpisode]:
        """逐episode从同一delta同时计算训练action和两种审查action。"""
        encoder = self.create_action_encoder()
        result = []
        for episode in episodes:
            representations = encoder.encode_episode_all(episode)
            result.append(
                EncodedEpisode(
                    episode=episode,
                    actions=representations.primary(self.config.action_encoding),
                    action_source=encoder.source,
                    action_encoding=self.config.action_encoding,
                    last_action_is_padding=encoder.last_action_is_padding,
                    euler_actions=representations.euler_control,
                    rotation_6d_actions=representations.rotation_6d_control,
                    auxiliary_fields=representations.auxiliary_fields,
                )
            )
        return result

    def run(self, preflight_only: bool = False) -> None:
        """执行主流程；preflight模式在模型加载和数据集写入之前返回。"""
        episodes, report = self.prepare(preflight_only=preflight_only)
        if not episodes:
            print(
                json.dumps(
                    {
                        "status": "already_complete",
                        "episodes": self.resume_state.source_episode_count,
                        "frames_written": self.resume_state.total_frames,
                        "output": str(self.config.output_dir),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return
        assert report is not None
        encoded_episodes = self.encode_actions(episodes)
        # 此处D由机器人类型和旋转表示共同决定：单/双臂Euler为7/14，6D为10/20。
        for index, item in enumerate(encoded_episodes):
            require_shape(
                f"episode[{index}].actions",
                item.actions,
                (item.episode.length, report.action_dimension),
            )
            trace_shape(f"ACTION episode={index}", actions=item.actions)
            arm_count = 2 if item.episode.is_dual_arm else 1
            require_shape(
                f"episode[{index}].euler_actions",
                item.euler_actions,
                (item.episode.length, 7 * arm_count),
            )
            require_shape(
                f"episode[{index}].rotation_6d_actions",
                item.rotation_6d_actions,
                (item.episode.length, 10 * arm_count),
            )
            if self.config.action_encoding.value == "cosmos_rotation_6d":
                required = set(AUXILIARY_ACTION_FIELDS + AUXILIARY_OBSERVATION_FIELDS)
                missing = sorted(required - set(item.auxiliary_fields))
                if missing:
                    raise ValueError(
                        "rotation-6D conversion requires aligned EE and joint fields; "
                        f"missing derived representations {missing} in {item.episode.path}"
                    )
            for name, values in item.auxiliary_fields.items():
                require_shape(name, values, (item.episode.length, values.shape[1]))
        prepared_stats = prepare_dataset_stats(
            self.config.stats_path,
            self.config.stats_mode,
            self.config.action_encoding,
            report.action_dimension,
        )
        stats = prepared_stats.effective
        # effective stats可能由官方14D Euler stats适配为20D rotation-6D stats。
        for key, dimension in (
            ("actions_min", report.action_dimension),
            ("actions_max", report.action_dimension),
            ("proprio_min", report.proprio_dimension),
            ("proprio_max", report.proprio_dimension),
        ):
            require_shape(f"stats.{key}", stats[key], (dimension,))
        trace_shape(
            "STATS",
            actions_min=stats["actions_min"],
            actions_max=stats["actions_max"],
            proprio_min=stats["proprio_min"],
            proprio_max=stats["proprio_max"],
        )
        if self.config.stats_mode is StatsMode.GENERATED:
            validate_external_stats(
                stats,
                self.config.stats_path,
                robot_type=report.robot_type,
                action_source=self.config.action_source,
                action_encoding=self.config.action_encoding,
                action_scale=self.config.action_scale,
                action_dimension=report.action_dimension,
                proprio_dimension=report.proprio_dimension,
            )
        else:
            validate_official_stats(stats, report.action_dimension, report.proprio_dimension)
        stats_range_report = validate_or_warn_stats_ranges(
            stats, episodes, encoded_episodes, self.config.stats_mode
        )
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        print(json.dumps(self.inspect_actions(encoded_episodes[0]), indent=2, ensure_ascii=False))
        if preflight_only:
            return
        # 后端从此处开始加载Policy/VAE、构造latent并写入Parquet。
        total_written = CosmosBackend(self.config).run(
            encoded_episodes,
            report,
            prepared_stats,
            stats_range_report,
            resume_state=self.resume_state,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "episodes_written_this_run": len(encoded_episodes),
                    "frames_written_this_run": total_written,
                    "resumed_from_episode": self.resume_state.start_episode,
                    "total_episodes": self.resume_state.source_episode_count,
                    "output": str(self.config.output_dir),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
