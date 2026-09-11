#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Actor server runner for distributed HILSerl robot policy training.

This script implements the actor component of the distributed HILSerl architecture.
It executes the policy in the robot environment, collects experience,
and sends transitions to the learner server for policy updates.

Examples of usage:

- Start an actor server for real robot training with human-in-the-loop intervention:
```bash
python -m lerobot.scripts.rl.actor --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: The actor server requires a running learner server to connect to. Ensure the learner
server is started before launching the actor.

**NOTE**: Human intervention is key to HILSerl training. Press the upper right trigger button on the
gamepad to take control of the robot during training. Initially intervene frequently, then gradually
reduce interventions as the policy improves.

**WORKFLOW**:
1. Determine robot workspace bounds using `find_joint_limits.py`
2. Record demonstrations with `gym_manipulator.py` in record mode
3. Process the dataset and determine camera crops with `crop_dataset_roi.py`
4. Start the learner server with the training configuration
5. Start this actor server with the same configuration
6. Use human interventions to guide policy learning

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide

Cosmos + Video2World（policy.type=cosmos）：
- 动作、价值、未来图像由 cosmos_utils.get_action 一次给出（data_batch 与 Video2WorldInference._get_data_batch_input 同源）。
- 可选：在 Hydra 中设置 video2world_experiment、video2world_ckpt_path 以额外加载 Video2WorldInference.generate_vid2world（独立检查点，显存加倍）。
- 需设置 cosmos_task_instruction（或 policy_obs[\"task\"]）作为语言条件。
"""
import logging
import os
import re
import time
from functools import lru_cache
from queue import Empty
from tqdm import tqdm
import json
from hil_logger import HILLogger
import grpc
import torch
import yaml
from torch import nn
from torch.multiprocessing import Event, Queue
from collections import deque
print('before import opencv')
from lerobot.cameras import opencv  # noqa: F401
print('1')

from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
print('2')
from lerobot.robots import so100_follower  # noqa: F401
# from lerobot.scripts.rl.gym_manipulator import make_robot_env
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.transport import services_pb2, services_pb2_grpc
print('3')
from lerobot.transport.utils import (
    bytes_to_state_dict,
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
print('4')
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.queue import get_last_item_from_queue
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.transition import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
)
from lerobot.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    init_logging,
)

import numpy as np
import h5py
from omegaconf import OmegaConf
print('5')
import draccus
from make_env import make_env
print('6')
from rl_envs.shared_state import shared_state
import hydra
print('7')
import traceback
import sys
from hydra.core.hydra_config import HydraConfig
import cv2
import csv
import copy
# “无 GUI / 无 X server”环境解决：pynput 在 Xvfb 下 import 可能阻塞，需 NO_GUI=1 跳过
import os
keyboard = None
if os.environ.get("NO_GUI", "0") != "1":
    try:
        from pynput import keyboard
    except Exception as e:
        print(f"[WARN] keyboard disabled: {e}")
        keyboard = None

from safetensors.torch import load_file
# from libero.libero import benchmark
# from cosmos_policy.experiments.robot.libero.libero_utils import (
#     get_libero_env,
# )
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.libero.run_libero_eval import validate_config as validate_config_cosmos
from cosmos_policy.experiments.robot.cosmos_utils import (
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    get_model,
    get_action,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    save_rollout_video,
    save_rollout_video_with_future_image_predictions,
)
from cosmos_policy.experiments.robot.robot_utils import (
    get_image_resize_size,
)
from cosmos_policy.utils.utils import jpeg_encode_image, set_seed_everywhere



ACTOR_SHUTDOWN_TIMEOUT = 30


def prepare_observation(obs, resize_size, flip_images: bool = False):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs, flip_images)
    wrist_img = get_libero_wrist_image(obs, flip_images)
    

    # Prepare observations dict
    observation = {
        "primary_image": img,
        "wrist_image": wrist_img,
        "proprio": np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])),
    }

    return observation  # Return processed observation


def save_image(image: np.ndarray, save_path: str) -> None:
    """Save a numpy image array (H, W, C) in RGB uint8 format to disk.

    Args:
        image: numpy array of shape (H, W, C), dtype uint8, RGB channel order.
        save_path: destination file path (e.g. "/tmp/obs/primary.png").
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bgr_image)


def print_green(msg: str) -> None:
    """Used by update_policy_parameters; keep a safe default if color helpers are absent."""
    print(msg)


def on_press(key):
    try:
        # print('on_press')
        if str(key) == 'Key.scroll_lock':
            print("----------------set human intervention key to {}!----------------".format(shared_state.human_intervention_key))
            shared_state.human_intervention_key = not shared_state.human_intervention_key
            time.sleep(0.5)
        # if str(key) == 'Key.space' or str(key) == 'Key.pause':
        elif str(key) == 'Key.pause':
        #     print("----------------set terminate to true!----------------")
            shared_state.terminate = True
            time.sleep(0.5)
        elif str(key) == 'Key.space':
            print("----------------set terminate to false!----------------")
            shared_state.emergency_terminate = True
            time.sleep(0.5)
        # else:
        #     print('unknown key : ', key)
    except AttributeError:
        pass

if keyboard is not None:
    try:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
    except Exception as e:
        print(f"[WARN] keyboard listener disabled: {e}")


from lerobot.configs.default import DatasetConfig
@hydra.main(config_path="./cfg", config_name="config", version_base=None) 
def actor_cli(env_cfg):
    if "ur" in env_cfg.robot_config.robot_type:
        lerobot_config_path = "../../train_config_silri_ur.json"
    elif "franka" in env_cfg.robot_config.robot_type:
        lerobot_config_path = "../../train_config_silri_franka.json"
    elif env_cfg.robot_config.robot_type == "sim":
        lerobot_config_path = "../../train_config_silri_sim.json"
    else:
        raise ValueError(f"Invalid robot type: {env_cfg.robot_type}")
    with draccus.config_type("json"):
        if not env_cfg.fix_gripper:
            cfg = draccus.parse(TrainRLServerPipelineConfig, lerobot_config_path, args=[f"--policy.type={env_cfg.policy_type}", f"--policy.num_discrete_actions=2"])
        else:
            cfg = draccus.parse(TrainRLServerPipelineConfig, lerobot_config_path, args=[f"--policy.type={env_cfg.policy_type}"])

    # todo:加入cosmos_cfg并validate
    with draccus.config_type("json"):
        if env_cfg.policy_type == "cosmos":
            cosmos_config_path = "../../train_config_cosmos.json"
            cosmos_cfg = draccus.parse(PolicyEvalConfig, cosmos_config_path, args=[])
    
    if env_cfg.dataset is not None:
        dataset_obj = OmegaConf.to_object(env_cfg.dataset)
        cfg.dataset = DatasetConfig(**dataset_obj)
    else:
        cfg.dataset = None


    cfg.validate()
    validate_config_cosmos(cosmos_cfg)

    # todo:加入t5_text_embeddings_cache
    init_t5_text_embeddings_cache(cosmos_cfg.t5_text_embeddings_path)
    # todo:加载数据集统计信息，用于动作和关节状态的归一化
    dataset_stats = load_dataset_stats(cosmos_cfg.dataset_stats_path)

  
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True

    # Create logs directory to ensure it exists
    cfg.job_name = env_cfg.task_name

    if env_cfg.robot_config.robot_type == "sim":
        cfg.env.features["observation.state"].shape = [18]
        cfg.policy.input_features["observation.state"].shape = [18]
    else:
        cfg.env.features["observation.state"].shape = [14] if env_cfg.use_force else [8]
        cfg.policy.input_features["observation.state"].shape = [14] if env_cfg.use_force else [8]

    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")
    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")
    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event



    parameters_queue = Queue()
    interactions_queue = Queue()

    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process


    act_with_policy(
        cfg=cfg,
        cosmos_cfg=cosmos_cfg,
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        interactions_queue=interactions_queue,
        env_cfg=env_cfg,
        dataset_stats=dataset_stats,
    )
    logging.info("[ACTOR] Policy process joined")


    # 取消队列的join线程（避免阻塞）
    logging.info("[ACTOR] join queues")
    parameters_queue.cancel_join_thread()

    logging.info("[ACTOR] queues closed")


def send_interactions(
    cfg: TrainRLServerPipelineConfig,
    interactions_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    Sends interactions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Interaction Messages:
        - Contains useful statistics about episodic rewards and policy timings.
        - The message is serialized using `pickle` and sent to the learner.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_interactions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor interactions process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendInteractions(
            interactions_stream(
                shutdown_event, interactions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming interactions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Interactions process stopped")

def interactions_stream(
    shutdown_event: Event,
    interactions_queue: Queue,
    timeout: float,  # type: ignore
) -> services_pb2.Empty:
    while not shutdown_event.is_set():
        try:
            message = interactions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Interaction queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message,
            services_pb2.InteractionMessage,
            log_prefix="[ACTOR] Send interactions",
        )

    return services_pb2.Empty()


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    cosmos_cfg: PolicyEvalConfig,
    shutdown_event: any,  # Event,
    parameters_queue: Queue,
    interactions_queue: Queue,
    env_cfg: any,
    dataset_stats: dict,
):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.

    Args:
        cfg: Configuration settings for the interaction process.
        shutdown_event: Event to check if the process should shutdown.
        parameters_queue: Queue to receive updated network parameters from the learner.
    """
    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")

    logging.info("make_env online")

    run_dir = HydraConfig.get().runtime.output_dir

    if env_cfg.policy_type == "cosmos":
        online_env = make_env(env_cfg, fake_env=False, use_human_intervention=env_cfg.use_human_intervention, classifier=False, use_gripper_penalty=False, cfg=cfg)
        print("----------------------------->>> already get libero env! ----------------------------->>>")
    else:
        online_env = make_env(env_cfg, fake_env=False, use_human_intervention=env_cfg.use_human_intervention, classifier=True, use_gripper_penalty=False, cfg=cfg)

    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    ### Instantiate the policy in both the actor and learner processes
    ### To avoid sending a SACPolicy object through the port, we create a policy instance
    ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
    try:
        if env_cfg.policy_type == "cosmos":
            # t5_text_embeddings_path 在 CosmosPolicy.__init__ 里加载 T5 缓存时需要
            cfg.policy.t5_text_embeddings_path = cosmos_cfg.t5_text_embeddings_path
            cfg.policy.cosmos_config_path = cosmos_cfg.config_file
        policy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
        )
        initial_state_dict = policy.state_dict()
        assert isinstance(policy, nn.Module)

        if env_cfg.policy_type == "cosmos":
            model, cosmos_config = get_model(cosmos_cfg)
            print("type(model): ", type(model))
            print("=========================== load model successfully! ===========================")
        else:
            load_path = env_cfg.load_path
            print(f"load_path: {load_path}")
            if load_path is not None:
                try:
                    state_dict = load_file(os.path.join(load_path, "model.safetensors"))
                    policy.load_state_dict(state_dict, strict=False)
                    policy.encoder_actor = policy.actor.encoder
                    if hasattr(policy, "discrete_actor"):
                        policy.discrete_actor.encoder = policy.actor.encoder
                except Exception as e:
                    logging.error(f"Error loading policy: {e}")
                    exit(-1)

        policy = policy.eval()

    except Exception as e:
        print(f"Error creating policy: {e}")
        traceback.print_exc()
        return

    
    obs, info = online_env.reset()

    success_episode = 0
    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0    # 累计当前episode的奖励
    episode_total_steps = 0         # 当前episode的总步数

    policy_timer = TimerManager("Policy inference", log=False)

    time_step = 0
    episode = 0
    episode_length_list = []
    violation_rate_episode_list = []
    violation_episode = 0


    csv_file = open("evaluation_result.csv", "a")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "sum_reward_episode", "success"])
    print('----- start evaluation -----')
    success_list=[]

    # replay回放h5文件的图片
    replay_images = []
    replay_wrist_images = [] if cosmos_cfg.use_wrist_image else None

    future_image_predictions_list = []
    # 获取预期图像维度
    resize_size = get_image_resize_size(cosmos_cfg.model_family) # 和libero相关还是和policy相关
    action_queue = deque(maxlen=cosmos_cfg.num_open_loop_steps)
    # 保存最近一次 best-of-n 的未来图像，供 step%k 落盘；避免队列未刷新时 best_future_predictions 未定义
    last_best_future_predictions = None
    base_seed = cosmos_cfg.seed  # Used for seed switching (if applicable)
    # 先不管这个step的问题，只考虑step一步的action、future_image_predictions、value_prediction
    # max_steps = TASK_MAX_STEPS[cosmos_cfg.task_suite_name]  # max_steps是单个episode的最大步数，但是cfg.policy.online_steps是总的在线步数


    for interaction_step in range(cfg.policy.online_steps):
        print(f"Interaction step: {interaction_step}")
        start_time = time.perf_counter()
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down act_with_policy")
            return

        with policy_timer:
            if "cosmos" not in env_cfg.policy_type:
                policy_obs = make_policy_obs(obs, device, env_cfg.robot_config.robot_type)
                action, _ = policy.select_action(batch=policy_obs)
                action = action.squeeze(0).cpu().detach().numpy()
                step_action = copy.deepcopy(action)
                print(f"step_action: {step_action}")

                next_obs, reward, terminated, truncated, info = online_env.step(step_action)
                done = terminated or truncated 
            else:
               # policy_obs = prepare_observation(obs, resize_size, cosmos_cfg.flip_images)

                # 保存policy_obs的图片到指定路径
                print("obs['right'].shape: ", obs["right"].shape)
                print("obs['wrist'].shape: ", obs["wrist"].shape)
                if interaction_step % 1 == 0:
                    obs_img_dir = os.path.join(run_dir, "obs_images", f"step_{interaction_step:06d}")
                    save_image(obs["right"], os.path.join(obs_img_dir, "right_image.png"))
                    if replay_wrist_images is not None:
                        save_image(obs["wrist"], os.path.join(obs_img_dir, "wrist_image.png"))
                
                # best-of-n
                if len(action_queue) == 0:
                    best_actions = None
                    best_future_predictions = None
                    
                    num_queries = cosmos_cfg.num_queries_best_of_n
                    query_results = []
                    for query_idx in range(num_queries):
                        actions_by_depth = []
                        future_image_predictions_by_depth = []
                        value_predictions_by_depth = []
                        return_dict = {}
                        # Query model to get action
                        start_time = time.time()
              
                        # 获取动作
                        # get_action放到wrapper中
                        task_description = "insert USB into socket"
                        # task_description = "pick up the gray plush toy on pink plate and lift it"
                        print(f"task_description: {task_description}")
                        action_return_dict = get_action(
                            cosmos_cfg,
                            model,
                            dataset_stats, # 数据集统计信息，用于动作和关节状态的归一化
                            obs,
                            task_description,
                            seed=cosmos_cfg.seed + query_idx,
                            randomize_seed=cosmos_cfg.randomize_seed,
                            num_denoising_steps_action=cosmos_cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=not (
                                cosmos_cfg.ar_future_prediction or cosmos_cfg.ar_value_prediction or cosmos_cfg.ar_qvalue_prediction
                            ),
                        )
                        print("================================== already get action ==================================")
                        query_time = time.time() - start_time
                        logging.info(
                            f"Query {query_idx + 1}/{num_queries}: Action query time = {query_time:.3f} sec"
                        )

                        return_dict["actions"] = action_return_dict["actions"]
                        actions_by_depth.append(return_dict["actions"])
                        # 当使用自回归获取未来状态预测、价值预测和q-value价值预测时
                        # return_dict["future_image_predictions_by_depth"] = future_image_predictions_by_depth
                        # return_dict["value_predictions_by_depth"] = value_predictions_by_depth
                        # 当不使用自回归获取未来状态预测、价值预测和q-value价值预测时：
                        return_dict["value_prediction"] = action_return_dict["value_prediction"]
                        return_dict["future_image_predictions"] = action_return_dict["future_image_predictions"]
                        
                        return_dict["actions_by_depth"] = actions_by_depth
                        query_results.append(return_dict)
                        
                    # Print all value predictions
                    logging.info(f"step={interaction_step}: Current base seed: {base_seed}")
                    for query_idx, return_dict in enumerate(query_results):
                        predicted_value = return_dict["value_prediction"]
                        logging.info(
                            f"Query {query_idx + 1}/{num_queries} (seed {cfg.seed + query_idx}): Predicted value = {predicted_value:.4f}",
                        )
                    # Get dict: seed number -> (action chunk, future state, value)
                    # 将每个动作候选的 action、future_image_predictions、value_prediction 打包成一个元组，并存储在一个字典中，键为种子数，值为元组
                    seed_to_return_dict = {
                        cosmos_cfg.seed + query_idx: (
                            return_dict["actions"],
                            return_dict["future_image_predictions"],
                            return_dict["value_prediction"],
                        )
                        for query_idx, return_dict in enumerate(query_results)
                    }
                    # Get seed with highest value、
                    # 选择价值最高的动作（Best-of-N）
                    best_seed, best_return_dict = max(seed_to_return_dict.items(), key=lambda x: x[1][2])
                    
                    best_actions = best_return_dict[0]
                    action_queue.extend(best_actions)
                    best_future_predictions = best_return_dict[1]
                    last_best_future_predictions = best_future_predictions
                    best_value_predictions = best_return_dict[2]
                    # Use the best actions, future predictions, and value predictions found
                    # # 把最优动作加入队列
                    # action_queue.extend(best_actions)
                    # # 把最优未来状态预测加入列表
                    # future_image_predictions_list.append(best_future_predictions)
                    logging.info(f"step={interaction_step}: Selected seed {best_seed} with value = {best_value_predictions:.4f}, with action = {best_actions}")

            if interaction_step % 1 == 0 and last_best_future_predictions is not None:
                obs_img_dir = os.path.join(run_dir, "prediction_images", f"step_{interaction_step:06d}")
                save_image(
                    last_best_future_predictions["future_image"],
                    os.path.join(obs_img_dir, "future_image.png"),
                )
                if replay_wrist_images is not None:
                    save_image(
                        last_best_future_predictions["future_wrist_image"],
                        os.path.join(obs_img_dir, "future_wrist_image.png"),
                    )

            # step_action = best_actions[0].tolist()
            step_action = action_queue.popleft()
            print(f"t: {interaction_step}\t action: {step_action}")
            # 保存预测的action到txt文件中
            with open(os.path.join(run_dir, "prediction_actions.txt"), "a") as f:
                f.write(f"{interaction_step}\t{step_action}\n")
            # Execute action in environment
            print("======================== before step env.step =========================")
            # next_obs, reward, done, info = online_env.step(step_action) # 放到wrapper里面
            print(interaction_step, ': step_action:', step_action)
            # input('press enter to continue')
            next_obs, reward, terminated, truncated, info = online_env.step(step_action)
            print(interaction_step, ': reward:', reward, ': terminated:', terminated, ': truncated:', truncated)
            print("after step")
            if shared_state.emergency_terminate:
                terminated = True
                shared_state.emergency_terminate = False
            done = terminated or truncated
            

         
        # print(interaction_step, ': reward:', reward, 'terminated:', terminated, 'truncated:', truncated)

        sum_reward_episode += float(reward)
        # Increment total steps counter for intervention rate
        episode_total_steps += 1
        time_step += 1
        mask = 1 - int(done)
        info["mask"] = mask
        obs = next_obs
        if done:
            print("======================== already done ==========================")
            # policy_obs = prepare_observation(obs, resize_size, cosmos_cfg.flip_images)
            # 保存当前结束的图片到指定路径
            obs_img_dir = os.path.join(run_dir, "obs_images", f"step_{interaction_step:06d}")
            save_image(obs["right"], os.path.join(obs_img_dir, "primary_image.png"))
            if replay_wrist_images is not None:
                save_image(obs["wrist"], os.path.join(obs_img_dir, "wrist_image.png"))
            
            


            episode_length_list.append(time_step)
            violation_rate_episode_list.append(violation_episode / time_step)
            violation_episode = 0
            if info["succeed"]:
                success_episode += 1
            interactions_queue.put(
                python_object_to_bytes(
                    {
                        "success": terminated,
                        "episode": interaction_step,
                        "sum_reward_episode": sum_reward_episode,
                        "Interaction step": interaction_step,
                    }
                )
            )
            csv_writer.writerow([interaction_step, sum_reward_episode, terminated])
            success_list.append(int(terminated))

            logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}, success: {terminated}")
            policy_timer.reset()
            time_step = 0
            episode += 1
            if episode == 20:
            # if episode == 100:
                break
            # 重置当前episode的计数器
            sum_reward_episode = 0.0
            episode_total_steps = 0
            
            obs, info = online_env.reset()
            # 新 episode 必须清空 action chunk 队列，否则会继续执行上一 episode 末尾
            # 基于旧观测预测的动作（通常是接近零的 hold-still 动作）
            action_queue.clear()
            last_best_future_predictions = None
            print("already done one episode, reset the environment (action_queue cleared)")
            

    average_success = 0.0
    if success_list:
        average_success = sum(success_list) / len(success_list)
        csv_writer.writerow(["average", "", average_success])
        
    csv_file.close()
    print('----- end evaluation -----')
    print(f'success_episode: {success_episode}/100')
    print(f'average_episode_length: {sum(episode_length_list) / len(episode_length_list)}')
    print(f'average_violation_rate: {sum(violation_rate_episode_list) / len(violation_rate_episode_list)}')
    
    # 保存 success_episode 到文件
    if env_cfg.load_path is not None:
        # 从 load_path 提取文件名
        # 将路径中的特殊字符替换为下划线，作为文件名
        safe_filename = re.sub(r'[^\w\-_./]', '_', env_cfg.load_path)
        safe_filename = safe_filename.replace('/', '_').replace('\\', '_')
        # 如果路径太长，只取最后一部分
        if len(safe_filename) > 200:
            safe_filename = os.path.basename(env_cfg.load_path)
            safe_filename = re.sub(r'[^\w\-_.]', '_', safe_filename)
        
        # 创建保存目录
        save_dir = os.path.join(cfg.output_dir, "eval_success")
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存文件
        save_file = os.path.join(save_dir, f"eval_success_{env_cfg.policy_type}_{env_cfg.seed}.txt")
        with open(save_file, 'w') as f:
            f.write(f"success_episode: {success_episode}\n")
            f.write(f"total_episodes: {episode}\n")
            f.write(f"average_success_rate: {average_success}\n")
            f.write(f"episode_length_list: {episode_length_list}\n")
            f.write(f"average_episode_length: {sum(episode_length_list) / len(episode_length_list)}\n")
            f.write(f"violation_rate_episode_list: {violation_rate_episode_list}\n")
            f.write(f"average_violation_rate: {sum(violation_rate_episode_list) / len(violation_rate_episode_list)}\n")
            f.write(f"test_path: {env_cfg.load_path}\n")
        logging.info(f"Success episode saved to: {save_file}")
        print(f"Success episode saved to: {save_file}")
    interactions_queue.put(
        python_object_to_bytes(
            {
                # "Interaction step": 20,
                "Interaction step": 100,
                "eval_end": True,
            }
        )
    )

    shutdown_event.set()


def make_policy_obs(obs: dict, device: torch.device, robot_type: str) -> dict:
    # 先将numpy数组转换为Tensor，再调整维度顺序
    policy_obs = {}
    for keys in obs.keys():
        if "state" not in keys:
            img = torch.from_numpy(obs[keys]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
            new_key = "observation.images." + keys
            policy_obs[new_key] = img
        else:
            state = torch.from_numpy(obs[keys]).float().unsqueeze(0).to(device)
            new_key = "observation.state"
            policy_obs[new_key] = state
    return policy_obs


def establish_learner_connection(
    stub: services_pb2_grpc.LearnerServiceStub,
    shutdown_event: Event,  # type: ignore
    attempts: int = 30,
):
    """Establish a connection with the learner.

    Args:
        stub (services_pb2_grpc.LearnerServiceStub): The stub to use for the connection.
        shutdown_event (Event): The event to check if the connection should be established.
        attempts (int): The number of attempts to establish the connection.
    Returns:
        bool: True if the connection is established, False otherwise.
    """
    for _ in range(attempts):
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_learner_connection")
            return False

        # Force a connection attempt and check state
        try:
            logging.info("[ACTOR] Send ready message to Learner")
            if stub.Ready(services_pb2.Empty()) == services_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for Learner to be ready... {e}")
            time.sleep(2)
    return False


@lru_cache(maxsize=1)
def learner_service_client(
    host: str = "127.0.0.1",
    port: int = 50051,
) -> tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]:
    """
    Returns a client for the learner service.

    GRPC uses HTTP/2, which is a binary protocol and multiplexes requests over a single connection.
    So we need to create only one client and reuse it.
    """

    print('host:', host, 'port:', port)

    channel = grpc.insecure_channel(
        f"{host}:{port}",
        grpc_channel_options(),
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    logging.info("[ACTOR] Learner service client created")
    return stub, channel


def receive_policy(
    cfg: TrainRLServerPipelineConfig,
    parameters_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
):
    """Receive parameters from the learner.

    Args:
        cfg (TrainRLServerPipelineConfig): The configuration for the actor.
        parameters_queue (Queue): The queue to receive the parameters.
        shutdown_event (Event): The event to check if the process should shutdown.
    """
    logging.info("[ACTOR] Start receiving parameters from the Learner")
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_receive_policy_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor receive policy process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        iterator = learner_client.StreamParameters(services_pb2.Empty())
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters",
        )

    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received policy loop stopped")




def update_policy_parameters(policy, parameters_queue: Queue, device):
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    print_green(f'----------------bytes_state_dict ------------------------------------------')
    
    if bytes_state_dict is not None:
        print_green(f'bytes_state_dict is not None')
        logging.info("[ACTOR] Load new parameters from Learner.")
        state_dicts = bytes_to_state_dict(bytes_state_dict)
        # TODO: check encoder parameter synchronization possible issues:
        # 1. When shared_encoder=True, we're loading stale encoder params from actor's state_dict
        #    instead of the updated encoder params from critic (which is optimized separately)
        # 2. When freeze_vision_encoder=True, we waste bandwidth sending/loading frozen params
        # 3. Need to handle encoder params correctly for both actor and discrete_critic
        # Potential fixes:
        # - Send critic's encoder state when shared_encoder=True
        # - Skip encoder params entirely when freeze_vision_encoder=True
        # - Ensure discrete_critic gets correct encoder state (currently uses encoder_critic)

        # Load actor state dict
        actor_state_dict = move_state_dict_to_device(state_dicts["policy"], device=device)
        policy.actor.load_state_dict(actor_state_dict)

        # Load discrete critic if present
        if hasattr(policy, "discrete_critic") and "discrete_critic" in state_dicts:
            discrete_critic_state_dict = move_state_dict_to_device(
                state_dicts["discrete_critic"], device=device
            )
            policy.discrete_critic.load_state_dict(discrete_critic_state_dict)
            logging.info("[ACTOR] Loaded discrete critic parameters from Learner.")

        if hasattr(policy, "encoder_critic") and "encoder_critic" in state_dicts:
            encoder_critic_state_dict = move_state_dict_to_device(
                state_dicts["encoder_critic"], device=device
            )
            policy.encoder_critic.load_state_dict(encoder_critic_state_dict)
            logging.info("[ACTOR] Loaded encoder critic parameters from Learner.")

        # Load discrete actor if present
        if hasattr(policy, "discrete_actor") and "discrete_actor" in state_dicts:
            discrete_actor_state_dict = move_state_dict_to_device(
                state_dicts["discrete_actor"], device=device
            )
            policy.discrete_actor.load_state_dict(discrete_actor_state_dict)
            logging.info("[ACTOR] Loaded discrete actor parameters from Learner.")

#################################################
#  Utilities functions #
#################################################


def push_transitions_to_transport_queue(transitions: list, transitions_queue):
    """Send transitions to learner in smaller chunks to avoid network issues.

    Args:
        transitions: List of transitions to send
        message_queue: Queue to send messages to learner
        chunk_size: Size of each chunk to send
    """
    transition_to_send_to_learner = []
    for transition in transitions:

        tr = move_transition_to_device(transition=transition, device="cpu")
        for key, value in tr["state"].items():
            if torch.isnan(value).any():
                logging.warning(f"Found NaN values in transition {key}")

        transition_to_send_to_learner.append(tr)

    transitions_queue.put(transitions_to_bytes(transition_to_send_to_learner))


def get_frequency_stats(timer: TimerManager) -> dict[str, float]:
    """Get the frequency statistics of the policy.

    Args:
        timer (TimerManager): The timer with collected metrics.

    Returns:
        dict[str, float]: The frequency statistics of the policy.
    """
    stats = {}
    if timer.count > 1:
        avg_fps = timer.fps_avg
        p90_fps = timer.fps_percentile(90)
        logging.debug(f"[ACTOR] Average policy frame rate: {avg_fps}")
        logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {p90_fps}")
        stats = {
            "Policy frequency [Hz]": avg_fps,
            "Policy frequency 90th-p [Hz]": p90_fps,
        }
    return stats





def sanitize_info_for_transition(info: dict) -> dict:
    """Sanitize info to only include types supported by Transition.complementary_info.

    Allowed types per downstream consumer: torch.Tensor, float, int, bool.
    - np.ndarray -> torch.from_numpy(...)
    - list/tuple of numbers/bools -> torch.tensor([...])
    - np.bool_/np.integer/np.floating -> Python scalar via .item()
    - torch.Tensor -> kept as is
    Unsupported types are skipped with a warning to avoid runtime errors.
    """
    safe_info = {}
    for key, value in info.items():
        try:
            if isinstance(value, torch.Tensor):
                safe_info[key] = value
            elif isinstance(value, np.ndarray):
                # Convert arrays directly to tensor; device will be handled later
                safe_info[key] = torch.from_numpy(value)
            elif isinstance(value, (list, tuple)):
                # If it's a sequence of numbers/bools, convert to tensor
                if all(isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)) for v in value):
                    safe_info[key] = torch.tensor([v.item() if isinstance(v, (np.integer, np.floating, np.bool_)) else v for v in value])
                else:
                    logging.warning(f"Dropping complementary_info[{key}] due to unsupported list/tuple contents type: {type(value)}")
            elif isinstance(value, (np.bool_, np.integer, np.floating)):
                safe_info[key] = value.item()
            elif isinstance(value, (int, float, bool)):
                safe_info[key] = value
            else:
                logging.warning(f"Dropping complementary_info[{key}] of unsupported type: {type(value)}")
        except Exception as e:
            logging.warning(f"Failed to sanitize complementary_info[{key}] ({type(value)}): {e}")
    return safe_info


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.actor == "threads"


def load_hydra_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML 配置文件不存在：{path}")
    cfg = OmegaConf.load(path)  # 加载 Hydra 风格 YAML
    OmegaConf.resolve(cfg)  # 解析 defaults 和 @_global_，合并 robot_type 和 task
    return cfg



if __name__ == "__main__":

    try:
        actor_cli()
    except Exception as e:
        print(f"In actor.py: [{type(e).__name__}] {e!r}")
        traceback.print_exc()          # full stacktrace
        sys.exit(1)