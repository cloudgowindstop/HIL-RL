"""转换使用的枚举、用户配置和下游运行配置。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .conditioning.camera import CameraState
from .conditioning.episode_labeling import EpisodeOutcome
from .encoding.multi_gpu_vae import validate_encode_settings


class ActionEncoding(str, Enum):
    """末端旋转在单步action中的表示方式。"""

    LEGACY_EULER = "legacy_euler"
    ROTATION_6D = "cosmos_rotation_6d"


class ActionSource(str, Enum):
    """目标末端位姿的来源和时间对齐方式。"""

    PUPPET_NEXT_FRAME = "puppet_next_frame"
    MASTER_POSE_SAME_FRAME = "master_same_frame"
    MASTER_JOINT_FK_SAME_FRAME = "master_joint_fk_same_frame"


class StatsMode(str, Enum):
    """generated要求语义sidecar；official直接兼容collect_data_cosmos stats。"""

    GENERATED = "generated"
    OFFICIAL = "official"


@dataclass(frozen=True)
class ActionScale:
    """第一层机器人控制scale，单位依次为米、弧度和夹爪原始单位。"""

    translation_m: float = 0.02
    rotation_rad: float = 0.06
    gripper: float = 1.0

    def validate(self, encoding: ActionEncoding) -> None:
        if self.translation_m <= 0 or self.gripper <= 0:
            raise ValueError("translation and gripper scales must be positive")
        if encoding is ActionEncoding.LEGACY_EULER and self.rotation_rad <= 0:
            raise ValueError("Euler rotation scale must be positive")


@dataclass(frozen=True)
class ConversionConfig:
    """CLI层用户配置；crop字段允许None，表示采用camera state默认方案。"""

    input_dir: Path
    output_dir: Path
    task: str
    episode_outcome: EpisodeOutcome
    stats_mode: StatsMode
    dataset_stats: Path | None = None
    official_dataset_stats: Path | None = None
    action_encoding: ActionEncoding = ActionEncoding.ROTATION_6D
    action_source: ActionSource = ActionSource.PUPPET_NEXT_FRAME
    action_scale: ActionScale = ActionScale()
    kinematics_config: Path | None = None
    t5_embeddings: Path | None = None
    skip_t5: bool = False
    episode_manifest: Path | None = None
    max_episodes: int = 0
    encode_batch_size: int = 16
    encode_world_size: int = 1
    encode_device_ids: tuple[int, ...] | None = None
    batch_size: int = 1
    save_clean_restore_latent: bool = False
    cosmos_config_path: Path = Path("train_config_cosmos.json")
    image_size: int = 224
    use_jpeg_compression: bool = True
    trained_with_image_aug: bool = True
    camera_state: CameraState = CameraState.NORMAL
    # None表示采用camera-state默认值；显式值作为实验覆盖项保留。
    wrist_crop_mode: str | None = None
    wrist_crop_fraction: float | None = None
    wrist_left_center_offset_x: int | None = None
    wrist_right_center_offset_x: int | None = None
    head_crop_top_pixels: int | None = None
    fk_max_position_error_m: float = 0.005
    fk_max_rotation_error_deg: float = 1.0
    monitor_memory: bool = False
    memory_sample_interval: float = 2.0
    monitor_storage: bool = False
    storage_sample_interval: float = 5.0
    storage_probe_mib: int = 8
    resume: bool = False

    def validate(self) -> None:
        """在读取大数据或加载模型前验证路径、组合约束和数值范围。"""
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"input directory does not exist: {self.input_dir}")
        if not self.task.strip():
            raise ValueError("task must not be empty")
        selected = self.stats_path
        if not selected.is_file():
            raise FileNotFoundError(f"dataset stats do not exist: {selected}")
        if self.stats_mode is StatsMode.GENERATED and self.official_dataset_stats is not None:
            raise ValueError("generated mode does not accept official_dataset_stats")
        if self.stats_mode is StatsMode.OFFICIAL and self.dataset_stats is not None:
            raise ValueError("official mode does not accept dataset_stats")
        self.action_scale.validate(self.action_encoding)
        if self.action_source is ActionSource.MASTER_JOINT_FK_SAME_FRAME:
            if self.kinematics_config is None:
                raise ValueError("master_joint_fk_same_frame requires --kinematics-config")
            if not self.kinematics_config.is_file():
                raise FileNotFoundError(f"kinematics config does not exist: {self.kinematics_config}")
            if self.action_encoding is not ActionEncoding.ROTATION_6D:
                raise ValueError("master joint FK currently requires cosmos_rotation_6d")
        if not self.skip_t5 and (self.t5_embeddings is None or not self.t5_embeddings.is_file()):
            raise FileNotFoundError("provide --t5-embeddings or explicitly use --skip-t5")
        if self.episode_manifest is not None and not self.episode_manifest.is_file():
            raise FileNotFoundError(f"episode manifest does not exist: {self.episode_manifest}")
        if self.max_episodes < 0 or self.encode_batch_size < 1 or self.batch_size < 1:
            raise ValueError("invalid episode or batch-size setting")
        validate_encode_settings(self.encode_world_size, self.encode_device_ids)
        if self.image_size != 224:
            raise ValueError("the current Cosmos latent layout requires image_size=224")
        if self.camera_state is not CameraState.NORMAL:
            raise ValueError(f"unsupported camera state: {self.camera_state}")
        if self.wrist_crop_mode is not None and self.wrist_crop_mode not in {"center_width", "bottom", "none"}:
            raise ValueError("wrist_crop_mode must be center_width, bottom, or none")
        if self.wrist_crop_fraction is not None and not 0.0 < self.wrist_crop_fraction <= 1.0:
            raise ValueError("wrist_crop_fraction must be in (0, 1]")
        if self.head_crop_top_pixels is not None and self.head_crop_top_pixels < 0:
            raise ValueError("head_crop_top_pixels must be >= 0")
        if self.memory_sample_interval <= 0:
            raise ValueError("memory_sample_interval must be positive")
        if self.storage_sample_interval <= 0:
            raise ValueError("storage_sample_interval must be positive")
        if self.storage_probe_mib < 1:
            raise ValueError("storage_probe_mib must be >= 1")

    @property
    def stats_path(self) -> Path:
        """根据stats mode返回唯一生效的stats路径。"""
        path = (
            self.dataset_stats
            if self.stats_mode is StatsMode.GENERATED
            else self.official_dataset_stats
        )
        if path is None:
            flag = "--dataset-stats" if self.stats_mode is StatsMode.GENERATED else "--official-dataset-stats"
            raise ValueError(f"{flag} is required for stats mode {self.stats_mode.value}")
        return path


@dataclass(frozen=True)
class CosmosRuntimeConfig:
    """传给Cosmos模块的完全解析配置；所有crop和路径字段均已有确定值。"""

    task_description: str
    episode_outcome: str
    stats_mode: str
    dataset_stats_path: Path
    cosmos_config_path: Path
    t5_embeddings_path: Path | None
    skip_t5: bool
    image_size: int
    use_jpeg_compression: bool
    trained_with_image_aug: bool
    camera_state: str
    wrist_crop_mode: str
    wrist_crop_fraction: float
    wrist_left_center_offset_x: int
    wrist_right_center_offset_x: int
    head_crop_top_pixels: int
    action_scale: tuple[float, float, float]
    chunk_size: int
    action_encoding: str
    action_source: str
    kinematics_config_path: Path | None
    encode_batch_size: int
    encode_world_size: int
    encode_device_ids: tuple[int, ...] | None
    batch_size: int
    save_clean_restore_latent: bool
