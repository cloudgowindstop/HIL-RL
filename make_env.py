import traceback
import sys
import gymnasium as gym
# import gym_hil 

from gymnasium.utils import seeding
from omegaconf import OmegaConf

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def _find_env_wrapper(env, wrapper_cls):
    cur = env
    while cur is not None:
        if isinstance(cur, wrapper_cls):
            return cur
        cur = getattr(cur, "env", None)
    return None


def set_sim_wrapper_flags(env, use_expert_control=None, use_inputs_control=None, evaluation_robust_env=False):
    """Toggle sim env wrapper flags (e.g. reuse online_env for rope eval).

    """
    from gym_hil.wrappers.hil_wrappers import ExpertControlWrapper, InputsControlWrapper

    inputs = _find_env_wrapper(env, InputsControlWrapper)
    if inputs is not None:
        if use_expert_control is not None:
            inputs.use_expert_control = use_expert_control
        if use_inputs_control is not None:
            inputs.use_inputs_control = use_inputs_control
    expert = _find_env_wrapper(env, ExpertControlWrapper)
    if expert is not None and use_expert_control is not None:
        expert.enabled = use_expert_control

    flex = env.unwrapped  # FlexEnv
    if hasattr(flex, "dataset_config"):
        flex.dataset_config["headless"] = True

    if evaluation_robust_env:
        flex.action_noise_duration = 2

        
def make_env(config, fake_env, use_human_intervention, classifier=False, expert_cfg=None, use_gripper_penalty=False, evaluation_env=False, evaluation_robust_env=False, task_name=None, seed=None, cfg=None, cosmos_cfg=None):
    try:  
        if config.robot_config.robot_type == "sim":
            # Convert OmegaConf to regular dict to allow non-primitive types like expert_policy
            env_cfg = OmegaConf.to_container(config.env_cfg, resolve=True)
            env_cfg["expert_cfg"] = expert_cfg
            if evaluation_env:
                env_cfg['headless'] = True
                env_cfg['use_inputs_control'] = False
                env_cfg['use_expert_control'] = False
                if "reset_delay_seconds" in env_cfg:
                    env_cfg['reset_delay_seconds'] = 0.0

                if not evaluation_robust_env:
                    env_cfg['action_noise_duration'] = 0

            if "rope" in task_name:
                if seed is not None:
                    from dynamics.utils import set_seed
                    set_seed(seed)

                if evaluation_robust_env:
                    env_cfg['action_noise_duration'] = 2
                else:
                    env_cfg['action_noise_duration'] = 0


            env = gym.make(**env_cfg)
            # FlexEnv.reset 使用 Gymnasium 的 env.np_random；仅 set_seed() 不会影响绳子初始弯曲等采样。
            if task_name and "rope" in task_name and seed is not None:
                inner = env.unwrapped
                inner._np_random, inner.np_random_seed = seeding.np_random(int(seed))
        else:
            from rl_envs.base_env import BaseEnv
            from rl_envs.tienyi_env import TienYiEnv
            from rl_envs.wrappers import HumanIntervention, SERLObsWrapper, AugmentedObservationWrapper, PoseActionPadWrapper, CosmosWrapper, FakeHumanIntervention
            from rl_envs.reward_wrapper import MultiCameraBinaryRewardClassifierWrapper, GripperPenaltyWrapper
            from rl_envs.subgoal import SubgoalRewardClassifierWrapper
            
            if "tien" in config.robot_config.robot_type:
                env = TienYiEnv(config=config.robot_config, fake_env=fake_env)
            else: 
                env = BaseEnv(config=config.robot_config, fake_env=fake_env)
            
            if not fake_env and use_human_intervention:
                env = HumanIntervention(env)
            
            if fake_env and use_human_intervention:
                env = FakeHumanIntervention(env)

            if getattr(config, "use_force_wrapper", False):
                from rl_envs.wrappers import EndEffectorForceInterventionWrapper
                env = EndEffectorForceInterventionWrapper(env, config=config.robot_config)

            # !!!add augmented observation wrapper修正
            env = AugmentedObservationWrapper(env)
            env = SERLObsWrapper(env,proprio_keys=config.robot_config.proprio_keys, use_force=config.use_force)
            if classifier:
                if getattr(config, "use_subgoals", False):
                    env = SubgoalRewardClassifierWrapper(env, config.subgoals, cfg=cfg)
                else:
                    env = MultiCameraBinaryRewardClassifierWrapper(
                        env, config.robot_config.classifier_cfg, cfg=cfg
                    )
            if use_gripper_penalty:
                env = GripperPenaltyWrapper(env, penalty=config.robot_config.gripper_penalty)
            if config.policy_type == "cosmos":
                env = CosmosWrapper(
                    env,
                    proprio_keys=config.robot_config.proprio_keys,
                    use_force=config.use_force,
                    cosmos_cfg=cosmos_cfg,
                    task_description=getattr(config, "task_description", None),
                    t5_text_embeddings_path=getattr(config, "t5_text_embeddings_path", None),
                    dual_arm=config.robot_config.dual_arm,
                )
            # 最外层：policy compact → 按 pose_action_mask 补齐为全维，再交给内层（含 GripperPenalty）
            if getattr(config.robot_config, "control_mode", "pose") == "pose":
                env = PoseActionPadWrapper(env)

    except Exception as e:
        print_green(f"[{type(e).__name__}] {e!r}")
        traceback.print_exc()          # full stacktrace
        sys.exit(1)
    return env
