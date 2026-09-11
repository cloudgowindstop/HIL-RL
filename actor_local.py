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
"""
import logging
import os
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

from lerobot.cameras import opencv  # noqa: F401

from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy

from lerobot.robots import so100_follower  # noqa: F401
from lerobot.scripts.rl.gym_manipulator import make_robot_env
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    bytes_to_python_object,
    bytes_to_state_dict,
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
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
from pynput import keyboard
from omegaconf import OmegaConf
import draccus
from make_env import make_env
from rl_envs.shared_state import shared_state
import hydra
import traceback
import sys
from hydra.core.hydra_config import HydraConfig
import cv2


from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.libero.run_libero_eval import validate_config as validate_config_cosmos

predict_action_cnt = 0
predict_action_time = 0
step_cnt = 0
step_time = 0
action_cnt = 0
action_time = 0
obs_cnt = 0
obs_time = 0
#################################################
# Main entry point #
#################################################


def on_press(key):
    try:
        if str(key) == 'Key.scroll_lock':
            print("----------------set human intervention key to {}!----------------".format(shared_state.human_intervention_key))
            shared_state.human_intervention_key = not shared_state.human_intervention_key
            time.sleep(0.5)
        # if str(key) == 'Key.space' or str(key) == 'Key.pause':
        if str(key) == 'Key.pause':
        #     print("----------------set terminate to true!----------------")
            shared_state.terminate = True
            time.sleep(0.5)
        if str(key) == 'Key.space':
            print("----------------set terminate to false!----------------")
            shared_state.emergency_terminate = True
            time.sleep(0.5)
    except AttributeError:
        pass
try:
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
except Exception as e:
    print("error in keyboard listener:", e)
    exit(0)

from lerobot.configs.default import DatasetConfig




@hydra.main(config_path="./cfg", config_name="config", version_base=None) 
def actor_cli(env_cfg):
    # _bypass_proxy_for_grpc()
    if "ur" in env_cfg.robot_config.robot_type:
        lerobot_config_path = "../../train_config_silri_ur.json"
    elif "franka" in env_cfg.robot_config.robot_type:
        lerobot_config_path = "../../train_config_silri_franka.json"
    elif env_cfg.robot_config.robot_type == "sim":
        lerobot_config_path = "../../train_config_silri_sim.json"
    else:
        raise ValueError(f"Invalid robot type: {env_cfg.robot_type}")

    # cosmos 策略需要在解析 cfg 之前就切换到对应的配置文件，否则 cfg 仍会从 silri 配置加载
    if env_cfg.policy_type == "cosmos":
        lerobot_config_path = "../../train_config_cosmos_franka.json"

    with draccus.config_type("json"):
        if not env_cfg.fix_gripper:
            cfg = draccus.parse(TrainRLServerPipelineConfig, lerobot_config_path, args=[f"--policy.type={env_cfg.policy_type}", f"--policy.num_discrete_actions=2"])
        else:
            cfg = draccus.parse(TrainRLServerPipelineConfig, lerobot_config_path, args=[f"--policy.type={env_cfg.policy_type}"])
    
    # todo:加入cosmos_cfg并validate
    cosmos_cfg = None
    if env_cfg.policy_type == "cosmos":
        with draccus.config_type("json"):
            cosmos_config_path = "../../train_config_cosmos.json"
            cosmos_cfg = draccus.parse(PolicyEvalConfig, cosmos_config_path, args=[])
            validate_config_cosmos(cosmos_cfg)
    
    if env_cfg.dataset is not None:
        dataset_obj = OmegaConf.to_object(env_cfg.dataset)
        cfg.dataset = DatasetConfig(**dataset_obj)
    else:
        cfg.dataset = None


    cfg.validate()

    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True

    # Create logs directory to ensure it exists
    cfg.job_name = env_cfg.task_name

    
    if env_cfg.policy_type != "cosmos":
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

    cloud_host, cloud_port = get_cloud_endpoint(cfg)
    cloud_client, grpc_channel = cloud_service_client(host=cloud_host, port=cloud_port)


    logging.info("[ACTOR] Establishing connection with cloud")
    if not establish_cloud_connection(cloud_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with cloud")
        return

    if not use_threads(cfg):
        # If we use multithreading, we can reuse the channel
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with cloud established")


    observation_queue = Queue()
    action_queue = Queue()


    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process


    # 任务4：将observation_queue中的观测数据发送给云端
    observation_process = concurrency_entity(
        target=send_observation,
        args=(cfg, observation_queue, shutdown_event),
        daemon=True,
    )
    # 任务5：将action_queue中的动作数据发送给云端
    action_process = concurrency_entity(
        target=receive_action,
        args=(cfg, action_queue, shutdown_event),
        daemon=True,
    )

    # # 启动任务
    observation_process.start()
    action_process.start()

    act_with_policy(
        cfg=cfg,
        shutdown_event=shutdown_event,
        observation_queue=observation_queue,
        action_queue=action_queue,
        env_cfg=env_cfg,
        cosmos_cfg=cosmos_cfg,
    )
    logging.info("[ACTOR] Policy process joined")

    # 关闭队列（阻止新数据写入）
    logging.info("[ACTOR] Closing queues")
    observation_queue.close()
    action_queue.close()
    

    # 等待并发任务结束
    observation_process.join()
    logging.info("[ACTOR] Observation process joined")
    action_process.join()
    logging.info("[ACTOR] Action process joined")
    # 取消队列的join线程（避免阻塞）
    logging.info("[ACTOR] join queues")
    observation_queue.cancel_join_thread()
    action_queue.cancel_join_thread()
    logging.info("[ACTOR] queues closed")





def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: any,  # Event,
    observation_queue: Queue,
    action_queue: Queue,
    env_cfg: any,
    cosmos_cfg: any,
):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.

    Args:
        cfg: Configuration settings for the interaction process.
        shutdown_event: Event to check if the process should shutdown.
        observation_queue: Queue to send observation to the cloud.
        action_queue: Queue to send action to the cloud.
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
    hil_logger = HILLogger(log_path=os.path.join(run_dir,"hil_log"))

    online_env = make_env(
        env_cfg,
        # fake_env=False,
        fake_env=True,
        use_human_intervention=env_cfg.use_human_intervention,
        classifier=False,
        use_gripper_penalty=cfg.policy.use_gripper_penalty,
        cfg=cfg,
        cosmos_cfg=cosmos_cfg,
    )
    print("------------------------------>>> already make env! ------------------------------>>>")

    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    obs, info = online_env.reset()
    print("------------------------------>>> success reset env! ------------------------------>>>")
    # print("------------------------------>>> obs:", obs)
    # print("------------------------------>>> info:", info)
    # run_dir = HydraConfig.get().runtime.output_dir
    # bs_img_dir = os.path.join(run_dir, "obs_images", f"step_000000")
    # print("obs['state']:", obs["state"])
    # print("obs['state']:", obs["state"].shape)
    # save_image(obs["right"], os.path.join(bs_img_dir, "fake_env_image_right.png"))
    # save_image(obs["wrist"], os.path.join(bs_img_dir, "fake_env_image_wrist.png"))
    # exit(0)
    # 将初始观测和 info 发送到云端
    print("--------------> before push reset observation to transport queue! -------------->")
    push_reset_observation_to_transport_queue(obs, info, observation_queue)

    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0    # 累计当前episode的奖励
    episode_intervention = False   
    episode_intervention_steps = 0  
    episode_total_steps = 0         
    prev_intervene = False

    policy_timer = TimerManager("Policy inference", log=False)
    time_step = 0
    episode = 0

    try:

        for interaction_step in range(cfg.policy.online_steps):
            start_time = time.perf_counter()
            if shutdown_event.is_set():
                online_env.close()
                hil_logger.close()
                logging.info("[ACTOR] Shutting down act_with_policy")
                return

            with policy_timer:
                if env_cfg.robot_config.robot_type == "sim":
                    action = np.zeros(4)
                else:
                    action = np.zeros(7)

                # Policy output action
                action_start_time = time.perf_counter()
                policy_action = get_action_from_cloud(action_queue, device)
                print("------------------------------->>> get action from cloud")
                policy_action_time = time.perf_counter() - action_start_time
                global predict_action_cnt
                global predict_action_time
                predict_action_cnt += 1
                predict_action_time += policy_action_time
                print("--------------------------------- predict_action_cnt:", predict_action_cnt, "predict_action_time:", predict_action_time / predict_action_cnt)

                if policy_action is None:
                    print("--------------------------------- cloud action is None, use zero action")
                    policy_action = np.zeros(7)
                print("policy action:", policy_action)

                # Ensure policy_action is a NumPy array on CPU before assigning into NumPy action
                
                # policy_action, action_info = policy.select_action(batch=policy_obs)
                
                if isinstance(policy_action, torch.Tensor):
                    policy_action = policy_action.squeeze(0).cpu().detach().numpy()
                action[0:policy_action.shape[0]] = policy_action


                if env_cfg.freeze_actor:
                    if env_cfg.robot_config.robot_type == "sim":
                        action = np.array([0.0, 0.0, 0.0, 0.0])
                    else:
                        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

                        
                # Calculate the FPS of the last policy inference
                policy_fps = policy_timer.fps_last  
                # Check if the policy inference efficiency meets the requirements (if below the threshold, alarm)
                log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)
                
            step_start_time = time.perf_counter()
            print("---------------------------------->>> before step! ------------------------------>>>")
            next_obs, reward, terminated, truncated, info = online_env.step(action)

            step_end_time = time.perf_counter()
            global step_cnt
            global step_time
            step_cnt += 1
            step_time += step_end_time - step_start_time
            print("--------------------------------- step_cnt:", step_cnt, "step_time:", step_time / step_cnt)

            # 将一步交互得到的 transition 信息发送到云端
            push_information_start_time = time.perf_counter()
            push_information_to_transport_queue(
                next_obs=next_obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                observation_queue=observation_queue,
            )
            push_information_end_time = time.perf_counter()
            global obs_cnt
            global obs_time
            obs_cnt += 1
            obs_time += push_information_end_time - push_information_start_time
            print("--------------------------------- obs_cnt:", obs_cnt, "obs_time:", obs_time / obs_cnt)
            done = terminated or truncated
            time_step += 1

            obs = next_obs
            if done:
                logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")
                stats = get_frequency_stats(policy_timer)
                policy_timer.reset()
                time_step = 0
                # Reset the counters for the current episode
                sum_reward_episode = 0.0
                episode_intervention = False
                episode_intervention_steps = 0
                episode_total_steps = 0
                obs, info = online_env.reset()
                push_reset_observation_to_transport_queue(obs, info, observation_queue)

        #    # Add the time span check at the end of the loop
        #     current_time_span = hil_logger.update_time_span()
        #     if current_time_span >= env_cfg.max_train_time:  
        #     # if done:  
        #         logging.info(f"[ACTOR] Time span reached {current_time_span} seconds, shut down all processes.")
        #         # Send the training complete message to the learner
        #         try:
        #             interactions_queue.put(
        #                 python_object_to_bytes(
        #                     {
        #                         "training_complete": True,
        #                         "Interaction step": interaction_step,
        #                         "Time span": current_time_span,
        #                         "message": "Training completed due to time limit reached",
        #                     }
        #                 )
        #             )
        #             logging.info("[ACTOR] Sent training complete message to Learner")
        #             exit(0)
        #         except Exception as e:
        #             logging.error(f"[ACTOR] Failed to send training complete message: {e}")
        #         # Set the shutdown event, notify all processes to exit
        #         shutdown_event.set()
        #         # Save the current data
        #         if len(list_transition_to_send_to_learner) > 0:
        #             push_transitions_to_transport_queue(
        #                 transitions=list_transition_to_send_to_learner,
        #                 transitions_queue=transitions_queue,
        #             )
        #         break
            
            
            if cfg.env.fps is not None:
                dt_time = time.perf_counter() - start_time
                busy_wait(1 / cfg.env.fps - dt_time)

    except (KeyboardInterrupt, Exception) as e:
        print(f"In actor.py: [{type(e).__name__}] {e!r}")
        online_env.close()
        hil_logger.close()
        logging.info("[ACTOR] Actor process closed")
        exit(0)    
    finally:
        online_env.close()
        hil_logger.close()
        logging.info("[ACTOR] Actor process closed")

#################################################
#  Communication Functions - Group all gRPC/messaging functions  #
#################################################
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



def establish_cloud_connection(
    stub: services_pb2_grpc.AsyncInferenceStub,
    shutdown_event: Event,  # type: ignore
    attempts: int = 30,
):
    """Establish a connection with the cloud.

    Args:
        stub (services_pb2_grpc.AsyncInferenceStub): The stub to use for the connection.
        shutdown_event (Event): The event to check if the connection should be established.
        attempts (int): The number of attempts to establish the connection.
    Returns:
        bool: True if the connection is established, False otherwise.
    """
    for _ in range(attempts):
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_cloud_connection")
            return False

        # Force a connection attempt and check state
        try:
            logging.info("[ACTOR] Send ready message to cloud")
            if stub.Ready(services_pb2.Empty()) == services_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for cloud to be ready... {e}")
            time.sleep(2)
    return False


@lru_cache(maxsize=1)
def cloud_service_client(
    host: str = "127.0.0.1",
    port: int = 50051,
) -> tuple[services_pb2_grpc.AsyncInferenceStub, grpc.Channel]:
    """
    Returns a client for the cloud service.

    GRPC uses HTTP/2, which is a binary protocol and multiplexes requests over a single connection.
    So we need to create only one client and reuse it.
    """

    print('host:', host, 'port:', port)


    channel = grpc.insecure_channel(
        f"{host}:{port}",
        grpc_channel_options(),
    )
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    logging.info("[ACTOR] Cloud service client created")
    return stub, channel


def get_cloud_endpoint(cfg: TrainRLServerPipelineConfig) -> tuple[str, int]:
    alc = cfg.policy.actor_learner_config
    host = getattr(alc, "cloud_host", "127.0.0.1")
    port = getattr(alc, "cloud_port", 50056)
    return host, port


def receive_action(
    cfg: TrainRLServerPipelineConfig,
    action_queue: Queue,
    shutdown_event: Event,  # type: ignore
    cloud_client: services_pb2_grpc.AsyncInferenceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
):
    """Receive action from the cloud.

    Args:
        cfg (TrainRLServerPipelineConfig): The configuration for the actor.
        action_queue (Queue): The queue to receive the action.
        shutdown_event (Event): The event to check if the process should shutdown.
    """
    logging.info("[ACTOR] Start receiving action from the cloud")
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_receive_action_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor receive action process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    # Lazily create gRPC client/channel if not provided (new process)
    if grpc_channel is None or cloud_client is None:
        cloud_host, cloud_port = get_cloud_endpoint(cfg)
        cloud_client, grpc_channel = cloud_service_client(host=cloud_host, port=cloud_port)

    try:
        while not shutdown_event.is_set():
            try:
                actions_message = cloud_client.GetActions(services_pb2.Empty())
            except grpc.RpcError as e:
                logging.error(f"[ACTOR] gRPC error while receiving actions: {e}")
                break

            # Server can return an Empty message when no actions are available
            actions_bytes = getattr(actions_message, "data", None)
            if not actions_bytes:
                continue

            # Push raw bytes to the queue; they will be deserialized in get_action_from_cloud
            action_queue.put(actions_bytes)

    except Exception as e:
        logging.error(f"[ACTOR] Unexpected error in receive_action: {e}")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received action loop stopped")



def send_observation(
    cfg: TrainRLServerPipelineConfig,
    observation_queue: Queue,
    shutdown_event: any,  # Event,
    cloud_client: services_pb2_grpc.AsyncInferenceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    Sends observation to the cloud.

    This function continuously retrieves messages from the queue and processes:

    - Observation Data:
        - An observation is collected.
        - Observations are moved to the CPU and serialized using PyTorch.
        - The serialized data is wrapped in a `services_pb2.Observation` message and sent to the cloud.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_observation_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor observation process logging initialized")

    # Lazily create gRPC client/channel if not provided (new process)
    if grpc_channel is None or cloud_client is None:
        # cloud_client, grpc_channel = cloud_service_client(
        #     host=cfg.policy.actor_learner_config.learner_host,
        #     port=cfg.policy.actor_learner_config.learner_port,
        # )
        cloud_host, cloud_port = get_cloud_endpoint(cfg)
        cloud_client, grpc_channel = cloud_service_client(host=cloud_host, port=cloud_port)

    try:
        cloud_client.SendObservations(
            observation_stream(
                shutdown_event,
                observation_queue,
                cfg.policy.actor_learner_config.queue_get_timeout,
            )
        )
    except Exception as e:
        traceback.print_exc()
        logging.error(f"[ACTOR] gRPC error while sending observations: {e}")
        exit(-1)
    # except grpc.RpcError as e:
    #     logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming observations")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Observation process stopped")



def observation_stream(shutdown_event: Event, observation_queue: Queue, timeout: float) -> services_pb2.Empty:  # type: ignore
    while not shutdown_event.is_set():
        try:
            message = observation_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Observation queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message, services_pb2.Observation, log_prefix="[ACTOR] Send observations"
        )

    return services_pb2.Empty()

def transitions_stream(shutdown_event: Event, transitions_queue: Queue, timeout: float) -> services_pb2.Empty:  # type: ignore
    while not shutdown_event.is_set():
        try:
            message = transitions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Transition queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message, services_pb2.Transition, log_prefix="[ACTOR] Send transitions"
        )

    return services_pb2.Empty()

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


#################################################
#  Policy functions #
#################################################



def get_action_from_cloud(action_queue: Queue, device, timeout: float = 120.0):
    """Wait for cloud inference result. Cosmos denoising can take multiple seconds."""
    param_start_time = time.perf_counter()
    bytes_action = get_last_item_from_queue(action_queue, block=True, timeout=timeout)
    param_end_time = time.perf_counter()
    global action_cnt
    global action_time
    action_cnt += 1
    action_time += param_end_time - param_start_time
    print("--------------------------------- action_cnt:", action_cnt, "action_time:", action_time / action_cnt)
    if bytes_action is not None:
        logging.info("[ACTOR] Load new action from cloud.")
        action_dict = bytes_to_state_dict(bytes_action)
        action = move_state_dict_to_device(action_dict["action"], device=device)
        if isinstance(action, torch.Tensor):
            action = action.squeeze(0).cpu().numpy()
        return action
    return None


#################################################
#  Utilities functions #
#################################################

def move_observation_to_device(observation: dict, device: str = "cpu") -> dict:
    # todo : debug the observation move to device
    device = torch.device(device)
    non_blocking = device.type == "cuda"
    for keys in observation.keys():
        if "state" not in keys:
            img = torch.from_numpy(observation[keys]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
            new_key = "observation.images." + keys
            observation[new_key] = img.to(device, non_blocking=non_blocking)
        else:
            state = torch.from_numpy(observation[keys]).float().unsqueeze(0).to(device)
            new_key = "observation.state"
            observation[new_key] = state.to(device, non_blocking=non_blocking)

    return observation


def observations_to_bytes(observation: dict) -> bytes:
    import io
    buffer = io.BytesIO()
    torch.save(observation, buffer)
    return buffer.getvalue()

def move_reset_observation_to_device(payload: dict, device: str = "cpu") -> dict:
    device = torch.device(device)
    non_blocking = device.type == "cuda"

    def _to_device_if_tensor_or_array(x):
        if isinstance(x, torch.Tensor):
            return x.to(device, non_blocking=non_blocking)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        return x

    payload["obs"] = {key: _to_device_if_tensor_or_array(val) for key, val in payload["obs"].items()}

    info = payload.get("info")
    payload["info"] = {key: _to_device_if_tensor_or_array(val) for key, val in info.items()}

    return payload

def push_reset_observation_to_transport_queue(obs: dict, info: dict, observation_queue: Queue) -> None:
    """将 reset 得到的 obs 和 info 发送到云端。"""
    payload = {
        "type": "reset",
        "obs": obs,
        "info": info,
    }
    # tr = move_reset_observation_to_device(payload, device="cpu")

    try:
        observation_queue.put(python_object_to_bytes(payload))
    except Exception as e:
        logging.error(f"[ACTOR] gRPC error while sending observations: {e}")
        exit(-1)
    # raw = observation_queue.get()
    # data = bytes_to_python_object(raw)
    # print("--------------------------------- push_reset_observation_to_transport_queue", data)



def push_information_to_transport_queue(
    next_obs: dict,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
    observation_queue: Queue,
) -> None:
    """将一步交互得到的 next_obs, reward, done, info 发送到云端。"""
    payload = {
        "type": "step",
        "next_obs": next_obs,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
    }
    observation_queue.put(python_object_to_bytes(payload))

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


def log_policy_frequency_issue(policy_fps: float, cfg: TrainRLServerPipelineConfig, interaction_step: int):
    if policy_fps < cfg.env.fps:
        logging.warning(
            f"[ACTOR] Policy FPS {policy_fps:.1f} below required {cfg.env.fps} at step {interaction_step}"
        )


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

def save_image(image: np.ndarray, save_path: str) -> None:
    """Save a numpy image array (H, W, C) in RGB uint8 format to disk.

    Args:
        image: numpy array of shape (H, W, C), dtype uint8, RGB channel order.
        save_path: destination file path (e.g. "/tmp/obs/primary.png").
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bgr_image)


if __name__ == "__main__":

    try:
        actor_cli()
    except (KeyboardInterrupt, Exception) as e:
        print(f"In actor.py: [{type(e).__name__}] {e!r}")

        traceback.print_exc()          # full stacktrace
        sys.exit(1)