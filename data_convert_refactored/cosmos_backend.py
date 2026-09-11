"""连接数据准备层与Cosmos/VAE写入层。

本模块只接收已计算好的action，不重新选择action来源，也不执行FK。其职责是加载
Policy、执行第二层归一化、逐episode构造transition，并交给writer编码和落盘。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .preparation.actions import EncodedEpisode
    from .config import ConversionConfig, CosmosRuntimeConfig
    from .preparation.dataset_stats import PreparedDatasetStats
    from .preflight import PreflightReport


class CosmosBackend:
    """维护一次转换共享的Policy、stats、T5缓存和LeRobot writer。"""

    def __init__(self, config: ConversionConfig):
        self.config = config

    def _chunk_size(self) -> int:
        """从Cosmos配置读取动作时间窗；当前正式配置默认预测未来16步。"""
        import json

        path = self.config.cosmos_config_path.expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        with path.open(encoding="utf-8") as file:
            return int(json.load(file).get("chunk_size", 16))

    def _runtime_config(self) -> CosmosRuntimeConfig:
        """把用户配置解析成下游模块可直接使用、无可选crop字段的运行配置。"""
        from .conditioning.camera import resolve_camera_processing
        from .config import CosmosRuntimeConfig

        camera = resolve_camera_processing(self.config.camera_state)

        return CosmosRuntimeConfig(
            task_description=self.config.task,
            episode_outcome=self.config.episode_outcome.value,
            stats_mode=self.config.stats_mode.value,
            dataset_stats_path=self.config.stats_path,
            cosmos_config_path=self.config.cosmos_config_path,
            t5_embeddings_path=(
                self.config.t5_embeddings if self.config.t5_embeddings is not None else None
            ),
            skip_t5=self.config.skip_t5,
            image_size=self.config.image_size,
            use_jpeg_compression=self.config.use_jpeg_compression,
            trained_with_image_aug=self.config.trained_with_image_aug,
            camera_state=self.config.camera_state.value,
            wrist_crop_mode=self.config.wrist_crop_mode or camera.wrist_crop_mode,
            wrist_crop_fraction=(
                self.config.wrist_crop_fraction
                if self.config.wrist_crop_fraction is not None
                else camera.wrist_crop_fraction
            ),
            wrist_left_center_offset_x=(
                self.config.wrist_left_center_offset_x
                if self.config.wrist_left_center_offset_x is not None
                else camera.wrist_left_center_offset_x
            ),
            wrist_right_center_offset_x=(
                self.config.wrist_right_center_offset_x
                if self.config.wrist_right_center_offset_x is not None
                else camera.wrist_right_center_offset_x
            ),
            head_crop_top_pixels=(
                self.config.head_crop_top_pixels
                if self.config.head_crop_top_pixels is not None
                else camera.head_crop_top_pixels
            ),
            action_scale=(
                self.config.action_scale.translation_m,
                self.config.action_scale.rotation_rad,
                self.config.action_scale.gripper,
            ),
            chunk_size=self._chunk_size(),
            action_encoding=self.config.action_encoding.value,
            action_source=self.config.action_source.value,
            kinematics_config_path=(
                self.config.kinematics_config
                if self.config.kinematics_config is not None
                else None
            ),
            encode_batch_size=self.config.encode_batch_size,
            encode_world_size=self.config.encode_world_size,
            encode_device_ids=self.config.encode_device_ids,
            batch_size=self.config.batch_size,
            save_clean_restore_latent=self.config.save_clean_restore_latent,
        )

    def _load_t5_embedding(self):
        """按任务文本读取缓存embedding；--skip-t5时明确返回None。"""
        if self.config.skip_t5:
            return None
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_t5_embedding_from_cache,
            init_t5_text_embeddings_cache,
        )

        init_t5_text_embeddings_cache(str(self.config.t5_embeddings))
        return get_t5_embedding_from_cache(self.config.task)

    def run(
        self,
        encoded_episodes: list[EncodedEpisode],
        report: PreflightReport,
        prepared_stats: PreparedDatasetStats,
        stats_range_report: dict,
        resume_state=None,
    ) -> int:
        """逐episode完成归一化、conditioning、VAE编码和Parquet写入。

        `encoded_episodes.actions`进入本函数时形状为(T,D)，已做机器人控制scale，
        尚未做dataset min/max归一化。返回值为实际写入的总帧数。
        """
        if not encoded_episodes:
            raise ValueError("cannot initialize Cosmos backend without episodes")

        from .memory_monitor import (
            memory_event,
            set_memory_context,
            start_memory_monitor,
            stop_memory_monitor,
        )
        from .storage_monitor import (
            set_storage_context,
            start_storage_monitor,
            stop_storage_monitor,
            storage_event,
            storage_probe,
        )

        start_memory_monitor(
            self.config.monitor_memory,
            self.config.output_dir / "logs",
            self.config.memory_sample_interval,
        )
        start_storage_monitor(
            self.config.monitor_storage,
            self.config.output_dir,
            self.config.storage_sample_interval,
            self.config.storage_probe_mib,
        )
        memory_event("backend_run_start", episode_count=len(encoded_episodes))
        storage_event("backend_run_start", episode_count=len(encoded_episodes))

        from .preparation.actions import normalize_encoded_episodes
        from .preparation.proprio import normalized_proprio_for_episode
        from .dataset_writer import (
            build_audit_feature_shapes,
            build_output_features,
            encode_and_write_episode,
            save_conversion_metadata,
        )
        from .encoding.policy_loader import init_cosmos_policy
        from .conditioning.transition_builder import build_transition_list_for_episode
        from lerobot.policies.cosmos.modeling_cosmos import (
            rescale_action as collect_rescale_action,
        )

        # 延迟导入保证--preflight-only不触发Torch、Cosmos模型和LeRobot重依赖加载。
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from .resume import (
            begin_episode_transaction,
            commit_episode_transaction,
            open_dataset_for_append,
        )

        runtime_config = self._runtime_config()
        first = encoded_episodes[0]
        action_dim = first.action_dim
        proprio_dim = report.proprio_dimension
        for item in encoded_episodes:
            if item.action_dim != action_dim:
                raise ValueError(
                    f"mixed action dimensions: expected {action_dim}, got {item.action_dim} "
                    f"in {item.episode.path}"
                )

        robot_type = report.robot_type
        audit_feature_shapes = build_audit_feature_shapes(first)
        for item in encoded_episodes[1:]:
            item_shapes = build_audit_feature_shapes(item)
            if item_shapes != audit_feature_shapes:
                raise ValueError(
                    f"mixed audit/source schemas: {item_shapes} != {audit_feature_shapes} "
                    f"in {item.episode.path}"
                )
        dataset_stats = prepared_stats.effective
        detected_fps = report.source_fps
        policy, cosmos_cfg = init_cosmos_policy(
            runtime_config,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
        )
        memory_event("after_policy_init")
        # 严格复用CosmosPolicy.normalizer()调用的同一模块级函数，完成第二层min/max归一化。
        encoded_episodes = normalize_encoded_episodes(
            encoded_episodes,
            dataset_stats,
            collect_rescale_action,
        )
        t5_embedding = self._load_t5_embedding()
        memory_event("after_t5_load")

        output_dir = self.config.output_dir
        repo_id = f"cosmos_{output_dir.name}"
        start_episode = resume_state.start_episode if resume_state is not None else 0
        if start_episode:
            dataset = open_dataset_for_append(output_dir, repo_id)
            expected_features = build_output_features(
                action_dim,
                proprio_dim,
                audit_feature_shapes,
                action_chunk_size=runtime_config.chunk_size,
                save_clean_restore_latent=runtime_config.save_clean_restore_latent,
            )
            for key, feature in expected_features.items():
                existing = dataset.features.get(key)
                if existing is None or tuple(existing["shape"]) != tuple(feature["shape"]):
                    raise RuntimeError(
                        f"resume feature mismatch for {key}: existing={existing}, expected={feature}"
                    )
            if dataset.meta.fps != detected_fps:
                raise RuntimeError(
                    f"resume FPS mismatch: existing={dataset.meta.fps}, source={detected_fps}"
                )
        else:
            dataset = LeRobotDataset.create(
                repo_id,
                fps=detected_fps,
                features=build_output_features(
                    action_dim,
                    proprio_dim,
                    audit_feature_shapes,
                    action_chunk_size=runtime_config.chunk_size,
                    save_clean_restore_latent=runtime_config.save_clean_restore_latent,
                ),
                root=str(output_dir),
                use_videos=False,
            )
        memory_event("after_dataset_create")
        fk_validation = report.fk
        if start_episode == 0:
            save_conversion_metadata(
                output_dir,
                runtime_config,
                robot_type,
                dataset_stats,
                list(report.cameras),
                detected_fps,
                fk_validation,
                dataset_stats,
                stats_range_report,
            )

        total_written = 0
        try:
            for local_index, item in enumerate(encoded_episodes):
                episode_index = start_episode + local_index
                # episode串行处理和写入；多GPU只在后续VAE microbatch内部生效。
                set_memory_context(
                    episode_index=episode_index,
                    episode_path=str(item.episode.path),
                    episode_frames=len(item.actions),
                )
                set_storage_context(
                    episode_index=episode_index,
                    episode_path=str(item.episode.path),
                    episode_frames=len(item.actions),
                )
                memory_event("episode_start")
                storage_event("episode_start")
                # Allocate and fsync real filesystem blocks before expensive VAE work.
                storage_probe("episode_start_probe")
                begin_episode_transaction(output_dir, episode_index, item.episode.path)
                normalized_proprio = normalized_proprio_for_episode(item.episode, dataset_stats)
                transitions, episode_info = build_transition_list_for_episode(
                    item.episode,
                    robot_type,
                    t5_embedding,
                    runtime_config,
                    normalized_actions=item.actions,
                    normalized_proprio=normalized_proprio,
                    euler_actions=item.euler_actions,
                    rotation_6d_actions=item.rotation_6d_actions,
                    auxiliary_fields=item.auxiliary_fields,
                )
                memory_event("after_build_transitions", transition_count=len(transitions))
                if episode_info["action_dim"] != action_dim:
                    raise ValueError(
                        f"transition action dim {episode_info['action_dim']} != {action_dim}"
                    )
                memory_event("before_encode_and_write")
                written = encode_and_write_episode(
                    transitions,
                    episode_index,
                    dataset,
                    policy,
                    cosmos_cfg,
                    runtime_config,
                    self.config.task,
                )
                total_written += written
                commit_episode_transaction(
                    output_dir,
                    episode_index,
                    item.episode.path,
                    written,
                    dataset.meta.total_frames,
                    resume_state.source_episode_count if resume_state is not None else len(encoded_episodes),
                )
                del transitions, episode_info, normalized_proprio
                memory_event("after_episode_release", written=written)
                memory_event("episode_complete", written=written)
            return total_written
        finally:
            stop_memory_monitor()
            stop_storage_monitor()
