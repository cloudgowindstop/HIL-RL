
import traceback
import sys
try:
    import numpy as np
    from scipy.spatial.transform import Rotation
    from make_env import make_env
    import hydra

    ACTOR_SHUTDOWN_TIMEOUT = 30
except Exception as e:
    print(f"Error in import: {e}")
    traceback.print_exc()
    sys.exit(1)
print("import success")


def _arm_names(env) -> list[str]:
    """记录哪些臂：free_* 时只记 active；双臂记 left+right。"""
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "active_arm_names"):
        return list(unwrapped.active_arm_names())
    if getattr(unwrapped, "dual_arm", False):
        return ["left", "right"]
    raise ValueError("Unknown arm names")


def _pose_to_xyz_euler(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).flatten()
    xyz = pose[:3]
    if pose.shape[0] >= 7:
        quat = pose[3:7]
        # currpos 正常为 xyz+quat；若范数明显不像单位四元数则当 euler
        if abs(np.linalg.norm(quat) - 1.0) < 0.35:
            euler = Rotation.from_quat(quat).as_euler("xyz")
        else:
            euler = pose[3:6]
    else:
        euler = pose[3:6]
    return np.hstack([xyz, euler]).astype(np.float64)


def _currpos_to_euler(env) -> np.ndarray:
    """固定维度：按 active/双臂臂列表取 xyz+euler，缺失填 nan。"""
    unwrapped = env.unwrapped
    currpos = unwrapped.currpos
    names = _arm_names(env)
    parts = []
    for name in names:
        if isinstance(currpos, dict):
            if name not in currpos or currpos[name] is None:
                parts.append(np.full(6, np.nan, dtype=np.float64))
            else:
                parts.append(_pose_to_xyz_euler(currpos[name]))
        else:
            parts.append(_pose_to_xyz_euler(currpos))
            break
    return np.concatenate(parts)


@hydra.main(config_path="./cfg", config_name="config", version_base=None)
def actor_cli(env_cfg):
    # valid_space 只依赖 hydra env_cfg + make_env，不需要解析 train_config_*.json
    print("before act_with_policy ...")
    act_with_policy(env_cfg=env_cfg)


def act_with_policy(env_cfg: any):
    image_keys = list(getattr(env_cfg.robot_config, "image_keys", []) or [])
    print(f"env cameras from robot_config.image_keys: {image_keys}")
    human_intervention = env_cfg.use_human_intervention
    online_env = make_env(
        env_cfg,
        fake_env=False,
        use_human_intervention=human_intervention,
        # classifier=True,
        classifier=False,
    )
    arm_names = _arm_names(online_env)
    time_step = 0
    print(f"recording arms: {arm_names} (each xyz+euler = 6 dims)")
    obs_keys = [k for k in online_env.observation_space.spaces.keys() if k != "state"]
    print(f"env observation image keys: {obs_keys}")
    if image_keys and set(obs_keys) != set(image_keys):
        print(
            f"WARN: observation image keys {obs_keys} != robot_config.image_keys {image_keys}"
        )

    gripper_pose = 0
    try:
        while True:
            online_env.reset()
            done = False
            all_pose = []
            while not done:
                if env_cfg.freeze_actor:
                    action = np.zeros(online_env.action_space.shape, dtype=np.float32)
                    # 仅全维 action（含 gripper）时翻转末维；compact 不含 gripper
                    if int(np.prod(online_env.action_space.shape)) >= 7:
                        action[-1] = gripper_pose
                        gripper_pose = 1.0 - gripper_pose
                else:
                    action = online_env.action_space.sample()
                next_obs, reward, terminated, truncated, info = online_env.step(action)
                print(f"time_step: {time_step}")
                time_step += 1
                done = terminated or truncated
                all_pose.append(_currpos_to_euler(online_env))

            all_pose = np.stack(all_pose, axis=0)
            all_pose_max = np.nanmax(all_pose, axis=0)
            all_pose_min = np.nanmin(all_pose, axis=0)
            # 按臂打印，便于对照 abs_pose_limit
            offset = 0
            for name in arm_names:
                print(f"{name}_pose_max (xyz+rpy):", all_pose_max[offset : offset + 6])
                print(f"{name}_pose_min (xyz+rpy):", all_pose_min[offset : offset + 6])
                offset += 6
            print("all_pose_max:", all_pose_max)
            print("all_pose_min:", all_pose_min)
    except (KeyboardInterrupt, Exception) as e:
        print(f"In actor.py: [{type(e).__name__}] {e!r}")
        online_env.close()
        logging.info("[ACTOR] Actor process closed")
        exit(0)
    finally:
        online_env.close()
        logging.info("[ACTOR] Actor process closed")


if __name__ == "__main__":
    try:
        actor_cli()
    except Exception as e:
        print(f"In actor.py: [{type(e).__name__}] {e!r}")
        traceback.print_exc()
        sys.exit(1)
