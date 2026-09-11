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
from concurrent.futures import ThreadPoolExecutor
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
    bytes_to_state_dict,
    bytes_to_python_object,
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
from omegaconf import OmegaConf
import draccus
import hydra
import traceback
import sys
from hydra.core.hydra_config import HydraConfig
import cv2

parms_cnt = 0
parms_time = 0
predict_action_cnt = 0
predict_action_time = 0
send_action_cnt = 0
send_action_time = 0
step_cnt = 0
step_time = 0
transition_cnt = 0
trajectory_cnt = 0
transition_time = 0

#################################################
# Main entry point #
#################################################


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

    learner_client, grpc_channel = learner_service_client(
        host="127.0.0.1",  # Learner 的 IP 地址
        port=6006,  # Learner 的端口号
        # host=cfg.policy.actor_learner_config.learner_host,  # Learner 的 IP 地址
        # port=cfg.policy.actor_learner_config.learner_port,  # Learner 的端口号
    )

    logging.info("[CLOUD ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[CLOUD ACTOR] Failed to establish connection with Learner")
        return

    if not use_threads(cfg):
        # If we use multithreading, we can reuse the channel
        grpc_channel.close()
        grpc_channel = None

    logging.info("[CLOUD ] Connection with Learner established")

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()
    observation_queue = Queue()
    action_queue = Queue()

    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process

    # 任务1：从learner接收模型参数，放入parameters_queue
    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    # 任务2：将transitions_queue中的过渡数据发送给learner
    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    # 任务3：将interactions_queue中的交互统计发送给learner
    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    # 启动与 learner 的通信进程
    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    # 启动 Cloud 侧的 AsyncInference gRPC 服务，供 local 连接
    async_server = grpc.server(
        ThreadPoolExecutor(max_workers=3),
        options=grpc_channel_options(),
    )
    async_servicer = CloudAsyncInferenceServicer(
        observation_queue=observation_queue,
        action_queue=action_queue,
        shutdown_event=shutdown_event,
    )
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(async_servicer, async_server)
    # cloud_host = os.environ.get("CLOUD_ACTOR_HOST", "0.0.0.0")
    # cloud_port = int(os.environ.get("CLOUD_ACTOR_PORT", "6006"))
    cloud_host = "0.0.0.0"
    cloud_port = 50056
    async_server.add_insecure_port(f"{cloud_host}:{cloud_port}")
    async_server.start()
    logging.info(f"[CLOUD ACTOR] AsyncInference server started on {cloud_host}:{cloud_port}")
    act_with_policy(
        cfg=cfg,
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        transitions_queue=transitions_queue,
        interactions_queue=interactions_queue,
        observation_queue=observation_queue,
        action_queue=action_queue,
        env_cfg=env_cfg,
    )
    logging.info("[ACTOR] Policy process joined")

    # 关闭队列（阻止新数据写入）
    logging.info("[ACTOR] Closing queues")
    transitions_queue.close()
    interactions_queue.close()
    parameters_queue.close()

    # 等待并发任务结束
    transitions_process.join()
    logging.info("[ACTOR] Transitions process joined")
    interactions_process.join()
    logging.info("[ACTOR] Interactions process joined")
    receive_policy_process.join()
    logging.info("[ACTOR] Receive policy process joined")

    # 关闭 AsyncInference server
    logging.info("[CLOUD ACTOR] Stopping AsyncInference server")
    async_server.stop(grace=None)
    # 取消队列的join线程（避免阻塞）
    logging.info("[ACTOR] join queues")
    transitions_queue.cancel_join_thread()
    interactions_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()
    observation_queue.cancel_join_thread()
    action_queue.cancel_join_thread()
    logging.info("[ACTOR] queues closed")


def get_observation_from_transport_queue(observation_queue: Queue, shutdown_event: Event, learner_client: services_pb2_grpc.AsyncInferenceStub | None = None) -> dict:
    """
    Get observation from the transport queue.
    """
    iterator = learner_client.SendObservations(services_pb2.Empty())
    return receive_bytes_in_chunks(
        iterator,
        observation_queue,
        shutdown_event,
        log_prefix="[CLOUD ACTOR] observation",
    )


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: any,  # Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
    observation_queue: Queue,
    action_queue: Queue,
    env_cfg: any,
):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.

    Args:
        cfg: Configuration settings for the interaction process.
        shutdown_event: Event to check if the process should shutdown.
        parameters_queue: Queue to receive updated network parameters from the learner.
        transitions_queue: Queue to send transitions to the learner.
        interactions_queue: Queue to send interactions to the learner.
    """
    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")


    run_dir = HydraConfig.get().runtime.output_dir
    hil_logger = HILLogger(log_path=os.path.join(run_dir,"hil_log"))


    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    ### Instantiate the policy in both the actor and learner processes
    ### To avoid sending a SACPolicy object through the port, we create a policy instance
    ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
    try:

        policy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
        )
        update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)
    except Exception as e:
        print(f"Error creating policy: {e}")
        return
    
    policy = policy.eval()
    assert isinstance(policy, nn.Module)

    # 从 local 收到的 reset 信息中获取 obs, info
    obs, info = get_initial_observation_from_transport_queue(observation_queue, shutdown_event)
    # print("--------------------------------- obs:", obs)
    # print("--------------------------------- info:", info)


    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0    # 累计当前episode的奖励
    list_transition_to_send_to_learner = []  
    episode_intervention = False   
    episode_intervention_steps = 0  
    episode_total_steps = 0         
    prev_intervene = False

    policy_timer = TimerManager("Policy inference", log=False)
    time_step = 0
    episode = 0

    for interaction_step in range(cfg.policy.online_steps):
        start_time = time.perf_counter()
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down act_with_policy")
            return

        with policy_timer:
            if env_cfg.robot_config.robot_type == "sim":
                action = np.zeros(4)
            else:
                action = np.zeros(7)

            # Policy output action
            policy_obs = make_policy_obs(obs, device, env_cfg.robot_config.robot_type)
            action_start_time = time.perf_counter()
            policy_action, action_info = policy.select_action(batch=policy_obs)
            policy_action = policy_action.squeeze(0).cpu().detach().numpy()
            policy_action_time = time.perf_counter() - action_start_time
            global predict_action_cnt
            global predict_action_time
            predict_action_cnt += 1
            predict_action_time += policy_action_time
            print("----------------------[ACTOR CLOUD] predict_action_cnt:", predict_action_cnt, "predict_action_time:", predict_action_time / predict_action_cnt)
            # print("--------------------------------- policy_action:", policy_action)
            
            action_start_time = time.perf_counter()
            push_action_to_transport_queue(policy_action, action_queue)
            action_end_time = time.perf_counter()
            global send_action_cnt
            global send_action_time
            send_action_cnt += 1
            send_action_time += action_end_time - action_start_time
            print("----------------------[ACTOR CLOUD] send_action_cnt:", send_action_cnt, "send_action_time:", send_action_time / send_action_cnt)
            action[0:policy_action.shape[0]] = policy_action

        step_start_time = time.perf_counter()
        next_obs, reward, terminated, truncated, info = get_infomation_from_local_actor_queue(
            observation_queue, shutdown_event
        )
        step_end_time = time.perf_counter()
        global step_cnt
        global step_time
        step_cnt += 1
        step_time += step_end_time - step_start_time
        print("----------------------[ACTOR CLOUD] get observation time:", step_cnt, "get observation time:", step_time / step_cnt)

        done = terminated or truncated

        sum_reward_episode += float(reward)
        # Increment the total steps counter for the intervention rate
        episode_total_steps += 1
        time_step += 1

        
        # NOTE: We override the action if the intervention is True, because the applied action is the intervention action
        if "is_intervention" in info and info["is_intervention"]:
            if not prev_intervene:
                if len(list_transition_to_send_to_learner) > 0:
                    list_transition_to_send_to_learner[-1]["complementary_info"]["first_intervene_reward"] = -1.0
            # NOTE: The action space for demonstration before hand is with the full action space
            # but sometimes for example we want to deactivate the gripper

            # print("is_intervention?", info["is_intervention"], info["intervene_action"])
            if env_cfg.robot_config.robot_type == "sim":
                action = info["teleop_action"]
            else:
                action = info["intervene_action"] 

            episode_intervention = True           
            # Increment intervention steps counter
            episode_intervention_steps += 1
        else:
            """
            恢复episode_intervention
            """
            episode_intervention = False

        prev_intervene = info["is_intervention"]
        hil_logger.log({"is_intervene": episode_intervention, "step": time_step, "episode": episode, "time": time.time(), "success": terminated})
        # print("time_step:", time_step, "current action:", action, 'reward:', reward)
        # 存储当前步的过渡数据
        obs_tensor = make_policy_obs(obs, device, env_cfg.robot_config.robot_type)
        next_obs_tensor = make_policy_obs(next_obs, device, env_cfg.robot_config.robot_type)
        act_tensor = torch.from_numpy(action)
        mask = 1 - int(done)
        info["mask"] = mask
        info["first_intervene_reward"] = 0

        list_transition_to_send_to_learner.append(
            Transition(
                state=obs_tensor,
                action=act_tensor,
                reward=reward,
                next_state=next_obs_tensor,
                done=terminated,
                truncated=truncated,  # TODO: (azouitine) Handle truncation properly
                complementary_info=sanitize_info_for_transition(info),
            )
        )
        # assign obs to the next obs and continue the rollout
        

        obs = next_obs
        # done = True
        if done:
            logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")

            # 更新网络参数
            update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)


            # 将当前episode收集的过渡数据推送到transitions_queu
            if len(list_transition_to_send_to_learner) > 0:

                push_transitions_start_time = time.perf_counter()
                push_transitions_to_transport_queue(
                    transitions=list_transition_to_send_to_learner,
                    transitions_queue=transitions_queue,
                )
                push_transitions_end_time = time.perf_counter()
                global transition_cnt
                global transition_time
                transition_cnt += len(list_transition_to_send_to_learner)
                global trajectory_cnt
                trajectory_cnt += 1
                transition_time += push_transitions_end_time - push_transitions_start_time
                print("----------------------[ACTOR CLOUD] send_transitions_to_learner_cnt:", transition_cnt, "send_transitions_time:", transition_time / transition_cnt, "trajectory_cnt:", trajectory_cnt, "trajectory_time:", transition_time / trajectory_cnt)
                
            list_transition_to_send_to_learner = []

            stats = get_frequency_stats(policy_timer)
            policy_timer.reset()

            # Calculate the intervention rate (intervention steps / total steps)
            intervention_rate = 0.0
            time_step = 0
            episode += 1
            if episode_total_steps > 0:
                intervention_rate = episode_intervention_steps / episode_total_steps
            # Send the episode statistics (reward, intervention rate, etc.) to the learner through interactions_queue
            interactions_queue.put(
                python_object_to_bytes(
                    {
                        "Episodic reward": sum_reward_episode,
                        "Interaction step": interaction_step,
                        "Episode intervention": int(episode_intervention),
                        "Intervention rate": intervention_rate,
                        **stats,
                    }
                )
            )


            # Reset the counters for the current episode
            sum_reward_episode = 0.0
            episode_intervention = False
            episode_intervention_steps = 0
            episode_total_steps = 0
            # obs, info = online_env.reset()
            obs, info = get_initial_observation_from_transport_queue(observation_queue, shutdown_event)

       # Add the time span check at the end of the loop
        current_time_span = hil_logger.update_time_span()
        if current_time_span >= env_cfg.max_train_time:  
        # if done:  
            logging.info(f"[ACTOR] Time span reached {current_time_span} seconds, shut down all processes.")
            # Send the training complete message to the learner
            try:
                interactions_queue.put(
                    python_object_to_bytes(
                        {
                            "training_complete": True,
                            "Interaction step": interaction_step,
                            "Time span": current_time_span,
                            "message": "Training completed due to time limit reached",
                        }
                    )
                )
                logging.info("[ACTOR] Sent training complete message to Learner")
                exit(0)
            except Exception as e:
                logging.error(f"[ACTOR] Failed to send training complete message: {e}")
            # Set the shutdown event, notify all processes to exit
            shutdown_event.set()
            # Save the current data
            if len(list_transition_to_send_to_learner) > 0:
                push_transitions_to_transport_queue(
                    transitions=list_transition_to_send_to_learner,
                    transitions_queue=transitions_queue,
                )
            break
        
        
        if cfg.env.fps is not None:
            dt_time = time.perf_counter() - start_time
            busy_wait(1 / cfg.env.fps - dt_time)


    hil_logger.close()

#################################################
#  Communication Functions - Group all gRPC/messaging functions  #
#################################################
def make_policy_obs(obs: dict, device: torch.device, robot_type: str) -> dict:
    # 先将numpy数组转换为Tensor，再调整维度顺序
    policy_obs = {}
    for keys in obs.keys():
        if "state" not in keys:
            val = obs[keys]
            if isinstance(val, torch.Tensor):
                img = val.permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
            else:
                img = torch.from_numpy(val).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
            # img = torch.from_numpy(obs[keys]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
            new_key = "observation.images." + keys
            policy_obs[new_key] = img
        else:
            val = obs[keys]
            if isinstance(val, torch.Tensor):
                state = val.float().unsqueeze(0).to(device)
            else:
                state = torch.from_numpy(val).float().unsqueeze(0).to(device)
            # state = torch.from_numpy(obs[keys]).float().unsqueeze(0).to(device)
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
    """step()
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
            host="127.0.0.1",
            port=6006,
            # host=cfg.policy.actor_learner_config.learner_host,
            # port=cfg.policy.actor_learner_config.learner_port,
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


def send_transitions(
    cfg: TrainRLServerPipelineConfig,
    transitions_queue: Queue,
    shutdown_event: any,  # Event,
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    Sends transitions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Transition Data:
        - A batch of transitions (observation, action, reward, next observation) is collected.
        - Transitions are moved to the CPU and serialized using PyTorch.
        - The serialized data is wrapped in a `services_pb2.Transition` message and sent to the learner.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_transitions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor transitions process logging initialized")

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host="127.0.0.1",
            port=6006,
            # host=cfg.policy.actor_learner_config.learner_host,
            # port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendTransitions(
            transitions_stream(
                shutdown_event, transitions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except Exception as e:
        traceback.print_exc()
        logging.error(f"[ACTOR] gRPC error: {e}")
        exit(-1)
    # except grpc.RpcError as e:
    #     logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming transitions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Transitions process stopped")


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
            host="127.0.0.1",
            port=6006,
            # host=cfg.policy.actor_learner_config.learner_host,
            # port=cfg.policy.actor_learner_config.learner_port,
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


class CloudAsyncInferenceServicer(services_pb2_grpc.AsyncInferenceServicer):
    """供 local actor 调用的异步推理服务，实现:
    - SendObservations: 接收 local 发来的 reset/step 信息，写入 observation_queue
    - GetActions: 从 action_queue 取出 cloud 计算好的 action 返回给 local
    """

    def __init__(self, observation_queue: Queue, action_queue: Queue, shutdown_event: Event) -> None:  # type: ignore[valid-type]
        self.observation_queue = observation_queue
        self.action_queue = action_queue
        self.shutdown_event = shutdown_event

    def Ready(self, request, context):  # noqa: N802
        logging.info("[CLOUD ACTOR] AsyncInference client ready")
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        # 当前场景下不需要远程配置 policy，直接忽略
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        logging.info("[CLOUD ACTOR] Receiving observations from local actor")
        receive_bytes_in_chunks(
            request_iterator,
            self.observation_queue,
            self.shutdown_event,
            log_prefix="[CLOUD ACTOR] observation",
        )
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        # 从队列中取出最近一次的 action bytes；如果当前没有，就返回空 Actions
        if self.shutdown_event.is_set():
            return services_pb2.Actions(data=b"")

        try:
            action_bytes = self.action_queue.get(timeout=0.1)
        except Empty:
            return services_pb2.Actions(data=b"")

        return services_pb2.Actions(data=action_bytes)


#################################################
#  Policy functions #
#################################################



def update_policy_parameters(policy, parameters_queue: Queue, device):
    param_start_time = time.perf_counter()
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    param_end_time = time.perf_counter()
    global parms_cnt
    global parms_time
    parms_cnt += 1
    parms_time += param_end_time - param_start_time
    print("----------------------[ACTOR CLOUD] parms_cnt:", parms_cnt, "parms_time:", parms_time / parms_cnt)
    if bytes_state_dict is not None:
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


def get_initial_observation_from_transport_queue(
    observation_queue: Queue,
    shutdown_event: Event,  # type: ignore[valid-type]
) -> tuple[dict, dict]:

    """从 observation_queue 中读取第一次 reset 信息，返回 obs, info。"""
    while not shutdown_event.is_set():
        if observation_queue.empty():
            time.sleep(0.01)
            continue

        raw = observation_queue.get()
        data = bytes_to_python_object(raw)
        if isinstance(data, dict) and data.get("type") == "reset":
            obs = data.get("obs", {})
            info = data.get("info", {}) or {}
            return obs, info

    return {}, {}


def get_infomation_from_local_actor_queue(
    observation_queue: Queue,
    shutdown_event: Event,  # type: ignore[valid-type]
) -> tuple[dict, float, bool, bool, dict]:
    """从 observation_queue 中读取一步 step 信息，返回 next_obs, reward, terminated, truncated, info。"""
    while not shutdown_event.is_set():
        if observation_queue.empty():
            time.sleep(0.01)
            continue

        raw = observation_queue.get()
        data = bytes_to_python_object(raw)
        if isinstance(data, dict) and data.get("type") == "step":
            next_obs = data.get("next_obs", {})
            reward = float(data.get("reward", 0.0))
            terminated = bool(data.get("terminated", False))
            truncated = bool(data.get("truncated", False))
            info = data.get("info", {}) or {}
            return next_obs, reward, terminated, truncated, info

    return {}, 0.0, False, False, {}


def push_action_to_transport_queue(action, action_queue: Queue) -> None:
    """将 cloud 计算得到的动作打包成 bytes，放入 action_queue，供 AsyncInference.GetActions 使用。"""
    import io

    if isinstance(action, np.ndarray):
        tensor = torch.from_numpy(action)
    elif isinstance(action, torch.Tensor):
        tensor = action.detach().cpu()
    else:
        tensor = torch.as_tensor(action)

    # 与 local 端 get_action_from_cloud 约定: state_dict 中 key 为 "action"
    buffer = io.BytesIO()
    torch.save({"action": tensor.unsqueeze(0)}, buffer)
    action_bytes = buffer.getvalue()
    action_queue.put(action_bytes)


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