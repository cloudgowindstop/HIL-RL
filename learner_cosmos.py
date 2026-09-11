# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
Learner server runner for distributed HILSerl robot policy training.

This script implements the learner component of the distributed HILSerl architecture.
It initializes the policy network, maintains replay buffers, and updates
the policy based on transitions received from the actor server.

Examples of usage:

- Start a learner server for training:
```bash
python -m lerobot.scripts.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: Start the learner server before launching the actor server. The learner opens a gRPC server
to communicate with actors.

**NOTE**: Training progress can be monitored through Weights & Biases if wandb.enable is set to true
in your configuration.

**WORKFLOW**:
1. Create training configuration with proper policy, dataset, and environment settings
2. Start this learner server with the configuration
3. Start an actor server with the same configuration
4. Monitor training progress through wandb dashboard

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import csv
import inspect
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat

import grpc
# from keras.src.callbacks import optimizer
import torch
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
# from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.scripts.rl import learner_service
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from lerobot.utils.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger
import hydra
import draccus
from omegaconf import OmegaConf
from lerobot.configs.default import DatasetConfig
import cv2
import numpy as np
from safetensors.torch import load_file

from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.config import load_config
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init
from cosmos_policy._src.imaginaire.utils import callback as imaginaire_callback
from cosmos_policy._src.imaginaire.utils import distributed as imaginaire_distributed
from cosmos_policy._src.imaginaire.utils.checkpointer import Checkpointer
from cosmos_policy._src.predict2.checkpointer.dcp import DistributedCheckpointer
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel
from cosmos_policy._src.imaginaire.utils import misc
from torch.utils.data import DataLoader, DistributedSampler
from megatron.core import parallel_state
from cosmos_policy._src.imaginaire.utils.misc import StragglerDetectorV2


LOG_PREFIX = "[LEARNER]"
expert_training_step = 0

new_offline_transition_num = 0

transition_cnt = 0
transition_time = 0
push_parameter_cnt = 0
push_parameter_time = 0

# Cosmos-only: same role as ``ImaginaireTrainer.checkpointer`` (save/load under ``cosmos_cfg.job`` / checkpoint).
cosmos_checkpointer: Checkpointer | None = None

#################################################
# MAIN ENTRY POINTS AND CORE ALGORITHM FUNCTIONS #
#################################################


# @parser.wrap()
@hydra.main(config_path="./cfg", config_name="config", version_base=None)
def train_cli(env_cfg):
    print("=======================in train_cli function=======================")
    if env_cfg.robot_config.robot_type == "ur_wrist":
        env_cfg.lerobot_config_path = "../../../../../train_config_silri_ur.json"
    elif "franka" in env_cfg.robot_config.robot_type:
        env_cfg.lerobot_config_path = "/media/HIL-RL-Project/HIL-RL/train_config_silri_franka_copy.json"
    elif env_cfg.robot_config.robot_type == "sim" :
        env_cfg.lerobot_config_path = "../../../../../train_config_silri_sim.json"
    else:
        raise ValueError(f"Invalid robot type: {env_cfg.robot_type}")
    
    
    config_path = env_cfg.lerobot_config_path

    # 使用 draccus.parse 直接加载配置，并通过 args 传入覆盖
    with draccus.config_type("json"):
        if not env_cfg.fix_gripper:
            cfg = draccus.parse(TrainRLServerPipelineConfig, config_path, args=[f"--policy.type={env_cfg.policy_type}",f"--policy.num_discrete_actions=2"])
        else:
            cfg = draccus.parse(TrainRLServerPipelineConfig, config_path, args=[f"--policy.type={env_cfg.policy_type}"])
    
    # 直接用cosmos的load_config函数加载配置，支持yaml和py文件
    if env_cfg.policy_type == "cosmos":
        cosmos_cfg = load_config(
            "cosmos_policy/config/config.py",
            ["--", "experiment=cosmos_predict2_2b_480p_libero"],
        )
        # print('cosmos_cfg:', cosmos_cfg) # cosmos_cfg: <cosmos_policy._src.imaginaire.config.Config object at 0x7f504fe81b10>
        print("-----------------already load cosmos config-----------------")
    print(os.path.join(cosmos_cfg.job.path_local, "checkpoints"))
    # exit(0)
    
    # Safely override dataset only if provided in env_cfg, converting Hydra DictConfig to DatasetConfig
    if hasattr(env_cfg, "dataset") and env_cfg.dataset is not None:
        try:
            dataset_obj = OmegaConf.to_object(env_cfg.dataset)
            if isinstance(dataset_obj, dict):
                cfg.dataset = DatasetConfig(**dataset_obj)
            else:
                cfg.dataset = dataset_obj
            
        except Exception as e:
            print(f"WARN: Ignoring invalid dataset override: {e}")
            exit(-1)

    # if env_cfg.policy_type == "cosmos":
    #     # 单机：DataLoader(dataset=..., shuffle=True)，不要 sampler。
    #     # 多卡：sampler=DistributedSampler(..., shuffle=False)，每个 epoch 调用 sampler.set_epoch(epoch)。
    #     dataset = instantiate(cosmos_cfg.dataloader_train.dataset)
    #     dataloader_train = DataLoader(
    #         dataset=dataset,
    #         shuffle=True,
    #         batch_size=cosmos_cfg.dataloader_train.batch_size,
    #         drop_last=cosmos_cfg.dataloader_train.drop_last,
    #         num_workers=cosmos_cfg.dataloader_train.num_workers,
    #         persistent_workers=cosmos_cfg.dataloader_train.persistent_workers,
    #         pin_memory=cosmos_cfg.dataloader_train.pin_memory,
    #         pin_memory_device=cosmos_cfg.dataloader_train.pin_memory_device,
    #         timeout=cosmos_cfg.dataloader_train.timeout,
    #     )
    #     print("-----------------already create dataloader_train-----------------")
    #     # print('dataloader_train:', dataloader_train) # dataloader_train: <torch.utils.data.dataloader.DataLoader object at 0x7f504fe81a90>

    cfg.output_dir = os.getcwd()
    cfg.job_name = env_cfg.task_name
    cfg.validate(config_path)
    cfg.wandb.name = env_cfg.task_name
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
    # Use the job_name from the config
    
    
    # if env_cfg.policy_type == "cosmos":
    #     model = instantiate(cosmos_cfg.model)
    #     trainer = cosmos_cfg.trainer.type(cosmos_cfg)
    #     trainer.train(model, dataloader_train, None)

    train(
        cfg,
        job_name=env_cfg.task_name,
        env_cfg=env_cfg,
        cosmos_cfg=cosmos_cfg,
    )   

    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None, env_cfg: any = None, cosmos_cfg: any = None):
    """
    Main training function that initializes and runs the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration
        job_name (str | None, optional): Job name for logging. Defaults to None.
    """
    print("----------------------> in train function")
    # cfg.validate()

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    try:
        # Setup WandB logging if enabled
        if cfg.wandb.enable and cfg.wandb.project:
            from lerobot.utils.wandb_utils import WandBLogger
            wandb_logger = WandBLogger(cfg)
        else:
            wandb_logger = None
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    except Exception as e:
        traceback.print_exc()
        exit(-1)

    # print("successfully create wandb logger...")
    # input("Press Enter to continue...")
    # Handle resume logic
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event
    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        env_cfg=env_cfg,
        cosmos_cfg=cosmos_cfg,
    )


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
    env_cfg: any,
    cosmos_cfg: any,
) -> None:
    """
    Start the learner threads for training.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        wandb_logger (WandBLogger | None): Logger for metrics
        shutdown_event: Event to signal shutdown
    """
    print("----------------------> in start_learner_threads function")
    # Create multiprocessing queues
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    # communication_process = concurrency_entity(
    #     target=start_learner,
    #     args=(
    #         parameters_queue,
    #         transition_queue,
    #         interaction_message_queue,
    #         shutdown_event,
    #         cfg,
    #     ),
    #     daemon=True,
    # )
    # communication_process.start()

    # 多个进程抢占同一个GRPC端口，导致无法启动
    # 所以需要使用torchrun多进程，每个进程都执行本函数，但是只有LOCAL_RANK==0的进程启动Actor通信进程
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        communication_process = concurrency_entity(
            target=start_learner,
            args=(
                parameters_queue,
                transition_queue,
                interaction_message_queue,
                shutdown_event,
                cfg,
            ),
            daemon=True,
        )
        communication_process.start()
    else:
        logging.info(
            "[LEARNER] LOCAL_RANK=%s: skip gRPC LearnerService (only rank 0 binds learner_port; torchrun multi-GPU).",
            local_rank,
        )
        communication_process = None

    add_actor_information_and_train(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        parameters_queue=parameters_queue,
        env_cfg=env_cfg,
        cosmos_cfg=cosmos_cfg,
    )
    logging.info("[LEARNER] Training process stopped")

    logging.info("[LEARNER] Closing queues")
    transition_queue.close()
    interaction_message_queue.close()
    parameters_queue.close()

    if communication_process is not None:
        communication_process.join()
        logging.info("[LEARNER] Communication process joined")

    logging.info("[LEARNER] join queues")
    transition_queue.cancel_join_thread()
    interaction_message_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[LEARNER] queues closed")


#######################################compute_target_prob##########
# Core algorithm functions #
#################################################


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
    env_cfg: any,
    cosmos_cfg: any,
):
    """
    Handles data transfer from the actor to the learner, manages training updates,
    and logs training progress in an online reinforcement learning setup.

    This function continuously:
    - Transfers transitions from the actor to the replay buffer.
    - Logs received interaction messages.
    - Ensures training begins only when the replay buffer has a sufficient number of transitions.
    - Samples batches from the replay buffer and performs multiple critic updates.
    - Periodically updates the actor, critic, and temperature optimizers.
    - Logs training statistics, including loss values and optimization frequency.

    NOTE: This function doesn't have a single responsibility, it should be split into multiple functions
    in the future. The reason why we did that is the  GIL in Python. It's super slow the performance
    are divided by 200. So we need to have a single thread that does all the work.

    Args:
        cfg (TrainRLServerPipelineConfig): Configuration object containing hyperparameters.
        wandb_logger (WandBLogger | None): Logger for tracking training progress.
        shutdown_event (Event): Event to signal shutdown.
        transition_queue (Queue): Queue for receiving transitions from the actor.
        interaction_message_queue (Queue): Queue for receiving interaction messages from the actor.
        parameters_queue (Queue): Queue for sending policy parameters to the actor.
    """
    print("----------------------> in add_actor_information_and_train function")
    # Extract all configuration variables at the beginning, it improve the speed performance
    # of 7%
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    online_step_before_learning = cfg.policy.online_step_before_learning
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint
    online_steps = cfg.policy.online_steps
    async_prefetch = cfg.policy.async_prefetch

    # 初始化cosmos模型，是否可以放在policy里面初始化？
    # cosmos1：定义初始化cosmos模型
    if env_cfg.policy_type == "cosmos":
        _ensure_cosmos_distributed_for_training(cosmos_cfg)
        cosmos_model = initialize_cosmos_model(cosmos_cfg) 
        print("----------------------> already create cosmos model")
    else:
        cosmos_model = None



    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Initialized logging for actor information and training process")

    logging.info("Initializing policy")


    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )


    assert isinstance(policy, nn.Module)

    policy.train()

    # # 暂时为了跑通这样修改
    # push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)

    last_time_policy_pushed = time.time()

    # cosmos2 : 初始化优化器、学习率调度器
    if env_cfg.policy_type != "cosmos":
        optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)
    else:
        optimizer, scheduler, grad_scaler = make_optimizers_and_scheduler_cosmos(cosmos_model=cosmos_model, cosmos_cfg=cosmos_cfg)
        print("----------------------> already create cosmos optimizer")

    # If we are resuming, we need to load the training state
    # cosmos3 : 加载训练状态，同步全局优化步数
    if env_cfg.policy_type != "cosmos":
        resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)
    else:
        resume_optimization_step, grad_accum_iter = load_training_state_cosmos(
            cosmos_cfg=cosmos_cfg,
            cosmos_model=cosmos_model,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
        )
        print("----------------------> already load cosmos training state")
    
    log_training_info(cfg=cfg, policy=policy)
    replay_buffer = initialize_replay_buffer(cfg, device, storage_device, cosmos_cfg)
    batch_size = cfg.batch_size
    offline_replay_buffer = None

    # cosmos6：初始化训练数据加载器/离线数据集
    if env_cfg.policy_type == "cosmos":
        dataloader_train = initialize_dataloader_train(cosmos_cfg)
        print("----------------------> already create cosmos dataloader")
    else:
        if cfg.dataset is not None:
            try:
                offline_replay_buffer = initialize_offline_replay_buffer(
                    cfg=cfg,
                    device=device,
                    storage_device=storage_device,
                    cosmos_cfg=cosmos_cfg,
                )
            except Exception as e:
                print(f"WARN: Ignoring invalid dataset override: {e}")
                exit(0)

        """
            只取offline_replay_buffer, batch_size无需减半
        """
        if "hgdagger" not in cfg.policy.type:
        # if not cfg.policy.only_off_and_intervention:
            batch_size: int = batch_size // 2  # We will sample from both replay buffer

    logging.info("Starting learner thread")
    interaction_message = None

    # cosmos4：初始化优化步数、交互步数，暂时没有交互步数
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    if env_cfg.policy_type != "cosmos":
        interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0
    else:
        interaction_step_shift = 0

    
    dataset_repo_id = None
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id

    # Initialize iterators
    online_iterator = None
    offline_iterator = None


    # =================== offline training ===================
    if "silri" in cfg.policy.type:
        offline_training(
            cfg=cfg,
            policy=policy,
            optimizers=optimizers,
            offline_replay_buffer=offline_replay_buffer,
            wandb_logger=wandb_logger,
        )
    # st_time = time.time()
    # NOTE: THIS IS THE MAIN LOOP OF THE LEARNER

    # cosmos9：初始化训练结束标志
    _end_training = False  # 训练结束标志：看起来和shutdown_event实现是一样的，是否可以合并？

    # Cosmos DataLoader：与 cosmos_policy/trainer.py 一致，每 epoch 调用 sampler.set_epoch 再打乱；
    # 迭代器跨 optimization_step 保持，耗尽后再进入下一 epoch（避免每步 iter 只重复第一个 batch）。
    cosmos_train_epoch = 0
    cosmos_dataloader_iter = None
    # 相邻 optimization_step 的 demo_sample_action_l1_loss 跳变超过阈值时，落盘前后两步的 batch 快照
    cosmos_demo_action_l1_spike_state = {"prev_loss": None, "prev_step": None, "prev_snap": None}
    
    # Initialize Straggler Detection
    straggler_detector = StragglerDetectorV2(
        enabled=cosmos_cfg.trainer.straggler_detection.enabled,
        report_freq=cosmos_cfg.trainer.straggler_detection.report_freq,
        profile_freq=cosmos_cfg.trainer.straggler_detection.profile_freq,
        max_diff=cosmos_cfg.trainer.straggler_detection.max_diff,
        raise_error=cosmos_cfg.trainer.straggler_detection.raise_error,
    )

    while True:
        # cosmos8：采样一个batch的数据，多卡则每个卡都采样一个batch的数据
        if env_cfg.policy_type == "cosmos":
            dataloader_train_iter= get_dataloader_train_iter(cosmos_train_epoch, dataloader_train)
            

        while True:
            # Exit the training loop if shutdown is requested
            if shutdown_event is not None and shutdown_event.is_set():
                logging.info("[LEARNER] Shutdown signal received. Exiting...")
                break

            # cosmos11：训练结束标志
            if env_cfg.policy_type == "cosmos":
                try:
                    # 获取一个batch的数据，精度对齐
                    forward_batch = get_forward_batch(cosmos_cfg, dataloader_train_iter, straggler_detector)
                    print("----------------------> already sample a batch of data")
                except StopIteration:
                    break  # 数据集已耗尽，进入下一epoch
                
                if optimization_step >= cosmos_cfg.trainer.max_iter:
                    _end_training = True
                    break
            # 暂时注释
            # # Process all available transitions to the replay buffer, send by the actor server
            # process_transitions(
            #     transition_queue=transition_queue,
            #     replay_buffer=replay_buffer,
            #     offline_replay_buffer=offline_replay_buffer,
            #     device=device,
            #     dataset_repo_id=dataset_repo_id,
            #     shutdown_event=shutdown_event,
            #     optimizers=optimizers,
            #     policy=policy,
            #     clip_grad_norm_value=clip_grad_norm_value,
            #     batch_size=batch_size,
            #     async_prefetch=async_prefetch,
            #     wandb_logger=wandb_logger,
            #     cfg=cfg,
            #     optimization_step=optimization_step
            # )

            # # Process all available interaction messages sent by the actor server
            # interaction_message = process_interaction_messages(
            #     interaction_message_queue=interaction_message_queue,
            #     interaction_step_shift=interaction_step_shift,
            #     wandb_logger=wandb_logger,
            #     shutdown_event=shutdown_event,
            # )
            
            # # Check if training is complete (message from actor)
            # if interaction_message is not None and interaction_message.get("training_complete", False):
            #     logging.info("[LEARNER] Received training complete message from Actor. Saving final checkpoint...")
            #     try:
            #         # cosmos7:保存检查点,没在优化critic的时候保存
            #         save_training_checkpoint(
            #             cfg=cfg,
            #             optimization_step=optimization_step,
            #             online_steps=online_steps,
            #             interaction_message=interaction_message,
            #             policy=policy,
            #             optimizers=optimizers,
            #             replay_buffer=replay_buffer,
            #             offline_replay_buffer=offline_replay_buffer,
            #             dataset_repo_id=dataset_repo_id,
            #             fps=fps,
            #         )
            #         logging.info("[LEARNER] Final checkpoint saved successfully")
            #     except Exception as e:
            #         logging.error(f"[LEARNER] Failed to save final checkpoint: {e}")
            #         traceback.print_exc()
            #     input("Press Enter to shut down all processes...")
            #     # Set shutdown event to exit training loop
            #     if shutdown_event is not None:
            #         shutdown_event.set()
            #         logging.info("[LEARNER] Shutdown event set due to training completion")
            #     break

            # # Wait until the replay buffer has enough samples to start training
            # if len(replay_buffer) < online_step_before_learning:
            #     continue

            # if online_iterator is None:
            #     online_iterator = replay_buffer.get_iterator(
            #         batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            #     )

            # if offline_replay_buffer is not None and offline_iterator is None:
            #     offline_iterator = offline_replay_buffer.get_iterator(
            #         batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            #     )

            # time_for_one_optimization_step = time.time()
            # 暂时注释
            
            # # 前 utd_ratio - 1 次优化critic
            # for _ in range(utd_ratio - 1):
            #     # Sample from the iterators
            #     """
            #         只需要离线数据+人类介入的数据
            #     """
            #     if "hgdagger" in cfg.policy.type:
            #     # if cfg.policy.only_off_and_intervention:
            #         if dataset_repo_id is not None:
            #             batch_offline = next(offline_iterator)
            #             batch_offline['is_intervention'] = torch.ones_like(batch_offline['done']).to(device)
            #             batch = batch_offline
            #     else:
                    
            #         batch = next(online_iterator)
            #         online_batch_size = batch["action"].shape[0]
            #         batch['is_intervention'] = batch["complementary_info"]["is_intervention"]
                
            #         if dataset_repo_id is not None:
            #             batch_offline = next(offline_iterator)
            #             batch_offline['is_intervention'] = torch.ones_like(batch_offline['done']).to(device)
            #             batch = concatenate_batch_transitions(
            #                 left_batch_transitions=batch, right_batch_transition=batch_offline
            #             )

            #     actions = batch["action"]
            #     rewards = batch["reward"]
            #     observations = batch["state"]
            #     next_observations = batch["next_state"]
            #     done = batch["done"]
            #     is_intervention = batch["is_intervention"]

            #     # weight = batch["complementary_info"]["weight"]

            #     assert 'discrete_penalty' in batch["complementary_info"].keys(), "discrete_penalty not in batch['complementary_info']"

            #     check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

            #     observation_features, next_observation_features = get_observation_features(
            #         policy=policy, observations=observations, next_observations=next_observations
            #     )

            #     # Create a batch dictionary with all required elements for the forward method
            #     forward_batch = {
            #         "action": actions,
            #         "reward": rewards,
            #         "state": observations,
            #         "next_state": next_observations,
            #         "done": done,
            #         "is_intervention": is_intervention,
            #         "observation_feature": observation_features,
            #         "next_observation_feature": next_observation_features,
            #         "complementary_info": batch["complementary_info"],
            #     }

            #     """
            #         hgdagger模仿学习不需要critic???
            #     """
            #     if "hgdagger" not in cfg.policy.type:
            #         # Use the forward method for critic loss
            #         critic_output = policy.forward(forward_batch, model="critic")

            #         # Main critic optimization
            #         loss_critic = critic_output["loss_critic"]
                    
            #         optimizers["critic"].zero_grad()
            #         loss_critic.backward()
            #         critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            #             parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            #         )
            #         optimizers["critic"].step()

            #     # Discrete critic optimization (if available)
            #     """
            #         hgdagger和lag夹爪部分模仿学习不需要discrete critic
            #     """
            #     if policy.config.num_discrete_actions is not None and "hgdagger" not in cfg.policy.type and "silri" not in cfg.policy.type:
            #         discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
            #         loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
            #         optimizers["discrete_critic"].zero_grad()
            #         loss_discrete_critic.backward()
            #         discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            #             parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
            #         )
            #         optimizers["discrete_critic"].step()
                    

            #     # Update target networks (main and discrete)
            #     policy.update_target_networks()

            
            # 暂时注释
            # # Sample for the last update in the UTD ratio
            # # 第 utd_ratio 次优化critic，同步更新 Actor
            # """
            #     只需要离线数据+人类介入的数据
            # """
            # if "hgdagger" in cfg.policy.type:
            # # if cfg.policy.only_off_and_intervention:
            #     if dataset_repo_id is not None:
            #         batch_offline = next(offline_iterator)
            #         batch_offline['is_intervention'] = torch.ones_like(batch_offline['done']).to(device)
            #         batch = batch_offline
            # else:
            #     batch = next(online_iterator)
            #     batch['is_intervention'] = batch["complementary_info"]["is_intervention"]
            
            #     if dataset_repo_id is not None:
            #         batch_offline = next(offline_iterator)
            #         batch_offline['is_intervention'] = torch.ones_like(batch_offline['done']).to(device)
            #         batch = concatenate_batch_transitions(
            #             left_batch_transitions=batch, right_batch_transition=batch_offline
            #         )

            # actions = batch["action"]
            # rewards = batch["reward"]
            # observations = batch["state"]
            # next_observations = batch["next_state"]
            # done = batch["done"]
            # is_intervention = batch["is_intervention"]


            # check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

            # observation_features, next_observation_features = get_observation_features(
            #     policy=policy, observations=observations, next_observations=next_observations
            # )

            # # Create a batch dictionary with all required elements for the forward method
            # forward_batch = {
            #     "action": actions,
            #     "reward": rewards,
            #     "state": observations,
            #     "next_state": next_observations,
            #     "done": done,
            #     "observation_feature": observation_features,
            #     "next_observation_feature": next_observation_features,
            #     "is_intervention": is_intervention,
            #     "complementary_info": batch["complementary_info"],
            # }

            # # cosmos8：采样一个batch的数据
            # if env_cfg.policy_type == "cosmos":
            #     dataloader_train.sampler.set_epoch(cosmos_train_epoch)
            #     dataloader_train_iter = iter(dataloader_train)
            #     forward_batch = next(dataloader_train_iter)
            #     forward_batch = misc.to(forward_batch, device="cuda")
            #     # 对齐混合精度：padding_mask 若保持 float32，会在 DiT 内 concat_padding_mask 时
            #     # 把输入提升到 float32，从而与 bf16 权重产生 dtype mismatch（float != bfloat16），无法矩阵相乘
            #     if isinstance(forward_batch, dict) and "padding_mask" in forward_batch:
            #         if "video" in forward_batch and isinstance(forward_batch["video"], torch.Tensor):
            #             forward_batch["padding_mask"] = forward_batch["padding_mask"].type_as(forward_batch["video"])
            #         else:
            #             forward_batch["padding_mask"] = forward_batch["padding_mask"].to(dtype=torch.bfloat16)
            #     print("----------------------> already sample a batch of data")

            """
                hgdagger模仿学习不需要critic???
            """
            # if "hgdagger" not in cfg.policy.type:
            #     critic_output = policy.forward(forward_batch, model="critic")

            #     loss_critic = critic_output["loss_critic"]

            #     optimizers["critic"].zero_grad()
            #     loss_critic.backward()
            #     critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            #         parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            #     )
            #     optimizers["critic"].step()

            #     # Initialize training info dictionary
            #     training_infos = {
            #         "loss_critic": loss_critic.item(),
            #         "critic_grad_norm": critic_grad_norm.mean().item(),
            #     }
            # else:
            training_infos = {
                "loss_actor": 0.0,
                "bc_loss": 0.0,
                "min_q_preds": 0.0,
                "actor_grad_norm": 0.0,
            }

            # # Discrete critic optimization (if available)
            # """
            #     hgdagger和silri部分模仿学习不需要discrete critic
            # """
            # if policy.config.num_discrete_actions is not None and "hgdagger" not in cfg.policy.type and "silri" not in cfg.policy.type:
            #     discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
            #     loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
            #     optimizers["discrete_critic"].zero_grad()
            #     loss_discrete_critic.backward()
            #     discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            #         parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
            #     ).item()
            #     optimizers["discrete_critic"].step()

            #     # Add discrete critic info to training info
            #     training_infos["loss_discrete_critic"] = loss_discrete_critic.item()
            #     training_infos["discrete_critic_grad_norm"] = discrete_critic_grad_norm
            #     training_infos["loss_q"] = discrete_critic_output["loss_q"].item()
            #     training_infos["loss_bc"] = discrete_critic_output["loss_bc"].item()
        
            
            # Actor and temperature optimization (at specified frequency)
            if optimization_step % policy_update_freq == 0:
                for _ in range(policy_update_freq):
                    # cosmos9:前向传播，获取loss，zero_grad，反向传播，优化器更新
                    # 拆开，分步执行
                    # 注意优化步数cosmos内部的优化步数
                    if env_cfg.policy_type == "cosmos":
                        with straggler_detector.profile_section(  # 检测慢节点（前向传播部分）
                            "fwd", cosmos_cfg.trainer.straggler_detection.analyze_forward,
                            profile_cuda=False,
                        ):
                            output_batch, loss = cosmos_model.training_step(forward_batch, iteration=optimization_step)
                            print("----------------------> already forward pass")
                            # # 与同一步号下的多次梯度更新对齐：仅最后一次 forward 的 loss 参与相邻 optimization_step 比较
                            # if _ == policy_update_freq - 1:
                            #     _maybe_save_cosmos_demo_action_l1_spike(
                            #         cfg=cfg,
                            #         forward_batch=forward_batch,
                            #         output_batch=output_batch,
                            #         optimization_step=optimization_step,
                            #         state=cosmos_demo_action_l1_spike_state,
                            #     )
                        with straggler_detector.profile_section(  # 检测慢节点（反向传播部分）
                            "bwd", cosmos_cfg.trainer.straggler_detection.analyze_backward,
                            profile_cuda=False,
                        ):
                            loss_scaled = grad_scaler.scale(loss / cosmos_cfg.trainer.grad_accum_iter)
                            loss_scaled.backward()
                            print("----------------------> already backward pass")
                        # 优化器更新新建一个函数update_optimizer_cosmos
                        update_optimizer_cosmos(
                            cosmos_cfg=cosmos_cfg, 
                            cosmos_model=cosmos_model, 
                            optimizer=optimizer, 
                            grad_scaler=grad_scaler, 
                            scheduler=scheduler, 
                            grad_accum_iter=grad_accum_iter, 
                            straggler_detector=straggler_detector, 
                            optimization_step=optimization_step)
                        print("----------------------> already update optimizer")
                        _maybe_log_cosmos_demo_l1_metrics(
                            output_batch=output_batch,
                            loss=loss,
                            optimization_step=optimization_step,
                            cosmos_train_epoch=cosmos_train_epoch,
                            wandb_logger=wandb_logger,
                            cfg=cfg,
                            cosmos_cfg=cosmos_cfg,
                        )
                    else:
                            # Actor optimization
                            actor_output = policy.forward(forward_batch, model="actor")
                            loss_actor = actor_output["loss_actor"] 
                            optimizers["actor"].zero_grad() # 重置Actor网络参数的梯度缓存
                            loss_actor.backward()
                            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                                    parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value
                                ).item()
                            optimizers["actor"].step()

                            # Add actor info to training info
                            training_infos["loss_actor"] = loss_actor.item()
                            training_infos["bc_loss"] = actor_output.get("bc_loss", torch.tensor(0.0)).item()
                            training_infos["min_q_preds"] = actor_output.get("min_q_preds", torch.tensor(0.0)).item()
                            training_infos["actor_grad_norm"] = actor_grad_norm
                            training_infos["allow_d_actor"] = actor_output.get("allow_d", 0)
                        

                            if "silri" in cfg.policy.type:
                                training_infos["lagrange_multiplier_value"] = actor_output["lagrange_multiplier_value"]
                            

                            if "silri" in cfg.policy.type and optimization_step % 1 == 0:
                                lagrange_output = policy.forward(forward_batch, model="lagrange")
                                loss_lagrange = lagrange_output["loss_lagrange"]

                                optimizers["lagrange"].zero_grad()
                                loss_lagrange.backward()
                                lagrange_grad_norm = torch.nn.utils.clip_grad_norm_(
                                    parameters=policy.lagrange_net.parameters(), max_norm=clip_grad_norm_value
                                ).item()
                                optimizers["lagrange"].step()
                                training_infos["loss_lagrange"] = loss_lagrange.item()
                                training_infos["lagrange_grad_norm"] = lagrange_grad_norm
                                training_infos["mean_d"] = lagrange_output["mean_d"]
                                training_infos["allow_d"] = lagrange_output["allow_d"]
                                training_infos["cost_dev"] = lagrange_output["cost_dev"]
                        
                            if env_cfg.policy_type != "cosmos":
                                policy.update_target_networks()

                            # # Temperature optimization
                            if "sac" in cfg.policy.type:
                                temperature_output = policy.forward(forward_batch, model="temperature")
                                loss_temperature = temperature_output["loss_temperature"]
                                optimizers["temperature"].zero_grad()
                                loss_temperature.backward()
                                temp_grad_norm = torch.nn.utils.clip_grad_norm_(
                                    parameters=[policy.log_alpha], max_norm=clip_grad_norm_value
                                ).item()
                                optimizers["temperature"].step()

                                # Add temperature info to training info
                                training_infos["loss_temperature"] = loss_temperature.item()
                                training_infos["temperature_grad_norm"] = temp_grad_norm
                                training_infos["temperature"] = policy.temperature

                                # Update temperature
                                policy.update_temperature()

            # # Push policy to actors if needed
            # # 将最新策略参数发送给 Actor 端，让 Actor 用新策略与环境交互
            # if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
            #     start_push_parameter_time = time.time()
            #     push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
            #     end_push_parameter_time = time.time()
            #     global push_parameter_cnt
            #     push_parameter_cnt += 1
            #     global push_parameter_time
            #     push_parameter_time += end_push_parameter_time - start_push_parameter_time
            #     # print(f"==========================================push parameter time: {push_parameter_time / push_parameter_cnt:.6f}s {push_parameter_cnt} push parameters")
            #     last_time_policy_pushed = time.time()

            # # Update target networks (main and discrete)

            # # Log training metrics at specified intervals
            # if optimization_step % 5 == 0:
            #     # print('-----------> training_infos:', training_infos)
            #     training_infos["replay_buffer_size"] = len(replay_buffer)
            #     if offline_replay_buffer is not None:
            #         training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
                    
            #     training_infos["Optimization step"] = optimization_step

            #     # Log training metrics
            #     if wandb_logger:
            #         # print('======================== logging training_infos with wandb logger ====================')
            #         wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

            # # Calculate and log optimization frequency
            # time_for_one_optimization_step = time.time() - time_for_one_optimization_step
            # frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)

            # # logging.info(f"[LEARNER] Optimization frequency loop [Hz]: {frequency_for_one_optimization_step}")

            # # Log optimization frequency
            # if wandb_logger:
            #     wandb_logger.log_dict(
            #         {
            #             "Optimization frequency loop [Hz]": frequency_for_one_optimization_step,
            #             "Optimization step": optimization_step,
            #         },
            #         mode="train",
            #         custom_step_key="Optimization step",
            #     )

            optimization_step += 1
            print("optimization_step:", optimization_step)


            # cosmos7:保存检查点
            if env_cfg.policy_type == "cosmos":
                if cosmos_checkpointer is not None and optimization_step % cosmos_cfg.checkpoint.save_iter == 0:
                    cosmos_checkpointer.save(
                        cosmos_model, optimizer, scheduler, grad_scaler, iteration=optimization_step
                    )
                    print("----------------------> already save cosmos checkpoint")
            else:
                if saving_checkpoint and (optimization_step % save_freq == 0 or optimization_step == online_steps):
                    print(f"Saving checkpoint at step {optimization_step}")
                    save_training_checkpoint(
                        cfg=cfg,
                        optimization_step=optimization_step,
                        online_steps=online_steps,
                        interaction_message=interaction_message,
                        policy=policy,
                        optimizers=optimizers,
                        replay_buffer=replay_buffer,
                        offline_replay_buffer=offline_replay_buffer,
                        dataset_repo_id=dataset_repo_id,
                        fps=fps,
                    )

        cosmos_train_epoch += 1
        print("cosmos_train_epoch:", cosmos_train_epoch)
        # 内层结束后的分流：非 cosmos 不应再套一层外层循环（内层即全部训练循环）
        if env_cfg.policy_type != "cosmos":
            break
        if _end_training:
            break
        if shutdown_event is not None and shutdown_event.is_set():
            break

    # cosmos7: 确保结束了也保存检查点
    if env_cfg.policy_type == "cosmos" and cosmos_checkpointer is not None:
        if optimization_step % cosmos_cfg.checkpoint.save_iter != 0:
            cosmos_checkpointer.save(
                cosmos_model, optimizer, scheduler, grad_scaler, iteration=optimization_step
            )
            print("----------------------> already save cosmos checkpoint")
        cosmos_checkpointer.finalize()
        print("----------------------> already finalize cosmos checkpoint")


def start_learner(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: any,  # Event,
    cfg: TrainRLServerPipelineConfig,
):
    """
    Start the learner server for training.
    It will receive transitions and interaction messages from the actor server,
    and send policy parameters to the actor server.

    Args:
        parameters_queue: Queue for sending policy parameters to the actor
        transition_queue: Queue for receiving transitions from the actor
        interaction_message_queue: Queue for receiving interaction messages from the actor
        shutdown_event: Event to signal shutdown
        cfg: Training configuration
    """
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_process_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Learner server process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        # Return back for MP
        # TODO: Check if its useful
        _ = ProcessSignalHandler(False, display_pid=True)

    service = learner_service.LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=cfg.policy.actor_learner_config.queue_get_timeout,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=learner_service.MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ],
    )

    services_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server,
    )

    # host = cfg.policy.actor_learner_config.learner_host
    host = "127.0.0.1"
    port = cfg.policy.actor_learner_config.learner_port

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(learner_service.SHUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")



import tqdm





def offline_training(cfg: TrainRLServerPipelineConfig, policy: nn.Module, optimizers: dict, offline_replay_buffer: ReplayBuffer, wandb_logger: WandBLogger | None):
    pretrain_dir = os.path.join(os.getcwd(), "../../../", f"expert_model_{cfg.policy.type}")
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)

    actor_pretrain_path = os.path.join(pretrain_dir, "actor_pretrain.pth")
    expert_pretrain_path = os.path.join(pretrain_dir, "expert_pretrain.pth")
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    if os.path.exists(expert_pretrain_path) and os.path.exists(actor_pretrain_path):
        # todo: debug
        policy.actor.load_state_dict(torch.load(actor_pretrain_path))
        policy.actor.train()
        policy.expert_network.load_state_dict(torch.load(expert_pretrain_path))
        policy.expert_network.train()

        if hasattr(policy, "actor_target"):
            policy.actor_target.load_state_dict(policy.actor.state_dict())
            policy.actor_target.eval()
        print(' success load actor pretrain model from ', actor_pretrain_path)
        return 


    batch_size = cfg.batch_size
    async_prefetch = cfg.policy.async_prefetch
    offline_iterator = None
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    # NOTE: THIS IS THE MAIN LOOP OF THE LEARNER
    offline_iterator = offline_replay_buffer.get_iterator(
            batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
        )
    global expert_training_step
    for optimization_step in tqdm.tqdm(range(500), desc="Offline training"):
        batch_offline = next(offline_iterator)
        batch_offline['is_intervention'] = torch.ones_like(batch_offline['done']).to(device)
        batch = batch_offline
        actions = batch["action"]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]
        is_intervention = batch["is_intervention"]


        check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

        observation_features, next_observation_features = get_observation_features(
            policy=policy, observations=observations, next_observations=next_observations
        )

        # Create a batch dictionary with all required elements for the forward method
        forward_batch = {
            "action": actions,
            "reward": rewards,
            "state": observations,
            "next_state": next_observations,
            "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
            "is_intervention": is_intervention,
            "complementary_info": batch["complementary_info"],
        }
    
        expert_output = policy.forward(forward_batch, model="expert")
        loss_expert = expert_output["loss_expert"] 
        optimizers["expert"].zero_grad()
        loss_expert.backward()
        expert_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.expert_network.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["expert"].step()
        actor_output = policy.forward(forward_batch, model="actor_bc")
        loss_actor_bc = actor_output["loss_actor_bc"] 
        # actor_output = policy.forward(forward_batch, model="actor")
        # loss_actor_bc = actor_output["loss_actor"]
        optimizers["actor"].zero_grad()
        loss_actor_bc.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["actor"].step()


        training_infos = {}
        # Add actor info to training info
        training_infos["loss_expert"] = loss_expert.item()
        training_infos["allow_d"] = expert_output.get("allow_d", 0)
        training_infos["loss_actor_bc"] = loss_actor_bc.item()


        # Log training metrics at specified intervals
        # if optimization_step % 10 == 0:
        training_infos["expert_training_step"] = expert_training_step
        if wandb_logger:
            wandb_logger.log_dict(d=training_infos, mode="expert", custom_step_key="expert_training_step")
        
        expert_training_step += 1



    # print('expert_pretrain_path:', expert_pretrain_path)
    torch.save(policy.expert_network.state_dict(), expert_pretrain_path)
    torch.save(policy.actor.state_dict(), actor_pretrain_path) 
    print(' success save actor pretrain model to ', actor_pretrain_path)
    # torch.save(policy.critic_ensemble.state_dict(), critic_pretrain_path)
    if hasattr(policy, "actor_target"):
        policy.actor_target.load_state_dict(policy.actor.state_dict())
        policy.actor_target.eval()



def expert_training(offline_replay_buffer, optimizers, policy, clip_grad_norm_value, device, async_prefetch, wandb_logger, optimization_step):
    batch_size = 256
    
    global expert_training_step

   
    offline_iterator = offline_replay_buffer.get_iterator(batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2)

    training_infos = {}
    print(' >>> begin expert update, expert_training_step:', expert_training_step)
    for _ in tqdm.tqdm(range(50), desc="Expert training"):
        batch_offline = next(offline_iterator)
        batch_offline['is_intervention'] = torch.ones_like(batch_offline['done']).to(device)
        batch = batch_offline
        actions = batch["action"]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]
        is_intervention = batch["is_intervention"]
        
        check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)
        observation_features, next_observation_features = get_observation_features(
            policy=policy, observations=observations, next_observations=next_observations
        )
        forward_batch = {
            "action": actions,
            "reward": rewards,
            "state": observations,
            "next_state": next_observations,
            "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
            "is_intervention": is_intervention,
            "complementary_info": batch["complementary_info"],
        }
        expert_output= policy.forward(forward_batch, model="expert")
        loss_expert = expert_output["loss_expert"] 
        allow_distance = expert_output.get("allow_d", 0)
        optimizers["expert"].zero_grad()
        loss_expert.backward()
        expert_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.expert_network.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["expert"].step()

        training_infos["loss_expert"] = loss_expert.item()
        training_infos["allow_d"] = allow_distance
        training_infos["expert_grad_norm"] = expert_grad_norm

        training_infos["expert_training_step"] = expert_training_step
        print(f'----------- in {expert_training_step} allow_d: {allow_distance}')

        if wandb_logger:
            wandb_logger.log_dict(d=training_infos, custom_step_key="expert_training_step", mode="expert")

        expert_training_step += 1


_COSMOS_DEMO_L1_METRIC_KEYS = (
    "demo_sample_action_l1_loss",
    "demo_sample_action_mse_loss",
    "demo_sample_future_proprio_l1_loss",
    "demo_sample_future_proprio_mse_loss",
    "world_model_sample_future_proprio_mse_loss",
    "world_model_sample_future_proprio_l1_loss",
    "demo_sample_future_wrist_image_l1_loss",
    "demo_sample_future_wrist_image_mse_loss",
    "world_model_sample_future_wrist_image_mse_loss",
    "world_model_sample_future_wrist_image_l1_loss",
    "demo_sample_future_image_l1_loss",
    "demo_sample_future_image_mse_loss",
    "world_model_sample_future_image_mse_loss",
    "world_model_sample_future_image_l1_loss",
    "demo_sample_value_l1_loss",
    "demo_sample_value_mse_loss",
    "world_model_sample_value_mse_loss",
    "world_model_sample_value_l1_loss",
)


def _tensor_to_log_float(t: torch.Tensor | None) -> float:
    if t is None:
        return float("nan")
    x = t.detach()
    if x.numel() == 0:
        return float("nan")
    if not x.is_floating_point():
        x = x.float()
    return float(x.mean().cpu())


# 阈值设置：当当前时刻的loss高于上一时刻THRESHOLD时，保存前后两步完整 batch
THRESHOLD = 0.2


def _first_batch_scalar_index(t: torch.Tensor | None) -> int:
    if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
        return -1
    return int(t.reshape(-1)[0].item())


def _clone_cosmos_forward_batch_cpu(forward_batch: dict) -> dict:
    """将当前 step 的 ``forward_batch`` 中可落盘内容克隆到 CPU（张量 detach clone，标量/ndarray 尽量拷贝）。"""
    snap: dict = {}
    for k, v in forward_batch.items():
        if isinstance(v, torch.Tensor):
            snap[k] = v.detach().cpu().clone()
        elif isinstance(v, dict):
            snap[k] = _clone_cosmos_forward_batch_cpu(v)
        elif isinstance(v, np.ndarray):
            snap[k] = v.copy()
        elif isinstance(v, (int, float, str, bool, type(None))):
            snap[k] = v
    return snap


def _extract_wrist_primary_action_views(forward_batch: dict) -> dict[str, torch.Tensor | None]:
    """从 ``video`` 时间维按 LIBERO 类 layout 切出当前 wrist / primary 条带；``action`` 使用 ``actions`` 键。"""
    out: dict[str, torch.Tensor | None] = {"wrist_image": None, "primary_image": None, "action": None}
    actions = forward_batch.get("actions")
    if isinstance(actions, torch.Tensor):
        out["action"] = actions.detach().cpu().clone()
    video = forward_batch.get("video")
    if not isinstance(video, torch.Tensor):
        return out
    aw = _first_batch_scalar_index(forward_batch.get("current_wrist_image_latent_idx"))
    ap = _first_batch_scalar_index(forward_batch.get("current_image_latent_idx"))
    aa = _first_batch_scalar_index(forward_batch.get("action_latent_idx"))
    num_dup = 4
    if aw >= 0 and ap > aw:
        num_dup = ap - aw
    elif ap >= 0 and aa > ap:
        num_dup = aa - ap
    tdim = video.shape[2]
    v = video.detach().cpu()
    if aw >= 0 and aw + num_dup <= tdim:
        out["wrist_image"] = v[:, :, aw : aw + num_dup, :, :].clone()
    if ap >= 0 and ap + num_dup <= tdim:
        out["primary_image"] = v[:, :, ap : ap + num_dup, :, :].clone()
    return out


def _output_batch_scalar_metrics(output_batch: dict) -> dict[str, float]:
    """只保留 ``output_batch`` 中小规模张量（避免 x0 / model_pred 等巨型张量）的标量摘要。"""
    out: dict[str, float] = {}
    for k, v in output_batch.items():
        if isinstance(v, torch.Tensor) and v.numel() <= 512:
            out[k] = _tensor_to_log_float(v)
    return out


def _tensor_to_txt_block(name: str, t: torch.Tensor | None, max_elems: int = 2048) -> str:
    if t is None:
        return f"{name}: <None>\n"
    x = t.detach().cpu().contiguous()
    lines = [f"{name}: shape={tuple(x.shape)} dtype={x.dtype} numel={x.numel()}\n"]
    if x.numel() == 0:
        return "".join(lines)
    xf = x.float() if not x.is_floating_point() else x
    lines.append(f"  min={float(xf.min()):.6g} max={float(xf.max()):.6g} mean={float(xf.mean()):.6g}\n")
    n = min(int(x.numel()), max_elems)
    flat = x.reshape(-1)[:n].numpy()
    lines.append(f"  first_{n}_elements (numpy repr):\n{np.array2string(flat, max_line_width=120)}\n")
    if x.numel() > n:
        lines.append(f"  ... ({x.numel() - n} more elements omitted)\n")
    return "".join(lines)


def _torch_tensor_to_numpy_saveable(t: torch.Tensor) -> np.ndarray:
    """``numpy.save`` 兼容：``bfloat16`` / ``float16`` 转为 float32，其余尽量 ``.numpy()``。"""
    tc = t.detach().cpu().contiguous()
    if tc.dtype in (torch.bfloat16, torch.float16):
        return tc.float().numpy()
    return tc.numpy()


def _save_chw_frame_preserve_dtype(chw: torch.Tensor, path_no_ext: str) -> str:
    """单帧 CHW 图像：``uint8`` 存 PNG（与训练一致的 RGB 通道序），其余 dtype 存 ``.npy``（``bf16`` 等为可读性转为 float32）。"""
    t = chw.detach().cpu().contiguous()
    if t.dtype == torch.uint8 and t.shape[0] == 3:
        arr = t.numpy()
        hwc = np.transpose(arr, (1, 2, 0))
        bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
        path = path_no_ext + ".png"
        cv2.imwrite(path, bgr)
        return path
    path = path_no_ext + ".npy"
    np.save(path, _torch_tensor_to_numpy_saveable(t))
    return path


def _save_bcthw_image_strip(t: torch.Tensor | None, spike_dir: str, stem: str) -> list[str]:
    """``[B,C,T,H,W]`` 腕部/主视角条带：按时间维逐帧落盘，保留各帧 dtype。"""
    if t is None or not isinstance(t, torch.Tensor):
        return []
    x = t.detach().cpu()
    if x.ndim != 5:
        path = os.path.join(spike_dir, f"{stem}_tensor.npy")
        np.save(path, _torch_tensor_to_numpy_saveable(x))
        return [os.path.basename(path)]
    b0 = x[0]
    c, ti, h, w = b0.shape
    written: list[str] = []
    for t_idx in range(ti):
        frame = b0[:, t_idx, :, :]
        path = _save_chw_frame_preserve_dtype(frame, os.path.join(spike_dir, f"{stem}_t{t_idx:02d}"))
        written.append(os.path.basename(path))
    return written


def _save_tensor_npy_if_present(t: torch.Tensor | None, path_no_ext: str) -> str | None:
    if t is None or not isinstance(t, torch.Tensor):
        return None
    path = path_no_ext + ".npy"
    np.save(path, _torch_tensor_to_numpy_saveable(t))
    return path


def _write_cosmos_demo_l1_spike_txt_bundle(
    spike_dir: str,
    *,
    threshold: float,
    prev_step: int,
    curr_step: int,
    prev_loss: float,
    curr_loss: float,
    prev_entry: dict,
    curr_entry: dict,
    pt_rel_path: str,
) -> str:
    """写入 ``report.txt``，并在子目录中保存 wrist/primary/action/proprio/value 等 sidecar 文件。"""
    report_lines: list[str] = [
        "=== demo_sample_action_l1_loss 相邻 optimization_step 跳变记录 ===\n",
        f"threshold: {threshold}\n",
        "说明: uint8 的 RGB 图像帧保存为 .png；其余张量保存为 .npy。"
        " torch.bfloat16 / float16 在 npy 内转为 float32 以便 numpy 读取；完整原始张量见 snapshot.pt。\n",
        f"prev_optimization_step: {prev_step}\n",
        f"curr_optimization_step: {curr_step}\n",
        f"prev_demo_sample_action_l1_loss: {prev_loss}\n",
        f"curr_demo_sample_action_l1_loss: {curr_loss}\n",
        f"loss_delta (curr - prev): {curr_loss - prev_loss}\n",
        f"abs_loss_delta: {abs(curr_loss - prev_loss)}\n",
        f"full_torch_snapshot_relative: {pt_rel_path}\n",
        "\n",
    ]

    def dump_one(tag: str, entry: dict) -> None:
        report_lines.append(f"--- {tag} optimization_step={entry.get('optimization_step')} ---\n")
        om = entry.get("output_metrics") or {}
        if om:
            report_lines.append("[output_batch 标量/小tensor 摘要 mean 值]\n")
            for kk in sorted(om.keys()):
                report_lines.append(f"  {kk}: {om[kk]}\n")
        report_lines.append(f"\n[{tag}] demo_sample_action_l1_loss: {entry.get('demo_sample_action_l1_loss')}\n\n")
        fb = entry.get("forward_batch_all_tensors_cpu") or {}
        report_lines.append(f"[{tag}] forward_batch 张量一览（shape / dtype）]\n")
        for ln in _forward_batch_tensor_tree_lines(fb):
            report_lines.append(ln + "\n")
        prefix = "prev" if tag == "PREV" else "curr"
        # 关键张量：原 dtype 存 npy；腕/主视角条带按帧 png 或 npy
        for key in ("actions", "proprio", "future_proprio", "value_function_return", "next_action_chunk", "next_value_function_return"):
            if key not in fb:
                continue
            v = fb[key]
            if isinstance(v, torch.Tensor):
                p = _save_tensor_npy_if_present(v, os.path.join(spike_dir, f"{prefix}_{key}"))
                if p:
                    report_lines.append(f"[{tag}] {key} -> {os.path.basename(p)}\n")
        wi = entry.get("wrist_image")
        paths = _save_bcthw_image_strip(wi, spike_dir, f"{prefix}_wrist_image")
        if paths:
            report_lines.append(f"[{tag}] wrist_image frames: {', '.join(paths)}\n")
        pi = entry.get("primary_image")
        paths_p = _save_bcthw_image_strip(pi, spike_dir, f"{prefix}_primary_image")
        if paths_p:
            report_lines.append(f"[{tag}] primary_image frames: {', '.join(paths_p)}\n")
        act = entry.get("action")
        if isinstance(act, torch.Tensor):
            p = _save_tensor_npy_if_present(act, os.path.join(spike_dir, f"{prefix}_action_views"))
            if p:
                report_lines.append(f"[{tag}] action (与 wrist/primary 对齐的 actions 视图) -> {os.path.basename(p)}\n")
        report_lines.append(f"\n[{tag}] 关键张量数值摘录\n")
        for key in ("actions", "proprio", "future_proprio", "value_function_return"):
            if key in fb and isinstance(fb[key], torch.Tensor):
                report_lines.append(_tensor_to_txt_block(f"{tag}.{key}", fb[key]))
        if isinstance(entry.get("action"), torch.Tensor):
            report_lines.append(_tensor_to_txt_block(f"{tag}.action_slice", entry["action"]))
        report_lines.append("\n")

    dump_one("PREV", prev_entry)
    dump_one("CURR", curr_entry)
    report_path = os.path.join(spike_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    return report_path


def _forward_batch_tensor_tree_lines(d: dict, prefix: str = "") -> list[str]:
    lines: list[str] = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, torch.Tensor):
            lines.append(f"  {prefix}{k}: shape={tuple(v.shape)} dtype={v.dtype}")
        elif isinstance(v, dict):
            lines.extend(_forward_batch_tensor_tree_lines(v, prefix + k + "."))
    return lines


def _maybe_save_cosmos_demo_action_l1_spike(
    *,
    cfg: TrainRLServerPipelineConfig,
    forward_batch: dict,
    output_batch: dict,
    optimization_step: int,
    state: dict,  # 上一时刻的loss和step
) -> None:
    if not imaginaire_distributed.is_rank0():
        return
    cur_loss = _tensor_to_log_float(output_batch.get("demo_sample_action_l1_loss"))
    full_snap = _clone_cosmos_forward_batch_cpu(forward_batch)
    views = _extract_wrist_primary_action_views(forward_batch)
    cur_entry = {
        "optimization_step": optimization_step,
        "demo_sample_action_l1_loss": cur_loss,
        "forward_batch_all_tensors_cpu": full_snap,
        "wrist_image": views["wrist_image"],
        "primary_image": views["primary_image"],
        "action": views["action"],
        "output_metrics": _output_batch_scalar_metrics(output_batch),
    }

    prev_loss = state.get("prev_loss")
    prev_step = state.get("prev_step")
    prev_snap = state.get("prev_snap")
    if prev_loss is not None and prev_snap is not None and prev_step is not None:
        if (cur_loss - prev_loss) > THRESHOLD:
            base_dir = os.path.join(cfg.output_dir, "logs", "demo_action_l1_spike")
            os.makedirs(base_dir, exist_ok=True)
            spike_name = f"spike_{int(prev_step):08d}_to_{int(optimization_step):08d}_dl{float(cur_loss - prev_loss):.4f}"
            spike_dir = os.path.join(base_dir, spike_name)
            os.makedirs(spike_dir, exist_ok=True)
            pt_name = "snapshot.pt"
            path = os.path.join(spike_dir, pt_name)
            torch.save(
                {
                    "threshold": THRESHOLD,
                    "loss_delta": float(cur_loss - prev_loss),
                    "prev": prev_snap,
                    "curr": cur_entry,
                },
                path,
            )
            report_path = _write_cosmos_demo_l1_spike_txt_bundle(
                spike_dir,
                threshold=THRESHOLD,
                prev_step=int(prev_step),
                curr_step=int(optimization_step),
                prev_loss=float(prev_loss),
                curr_loss=float(cur_loss),
                prev_entry=prev_snap,
                curr_entry=cur_entry,
                pt_rel_path=pt_name,
            )
            logging.info(
                "[LEARNER] demo_sample_action_l1_loss 跳变 %.4f -> %.4f (|Δ|=%.4f)，已保存目录: %s （report: %s, snapshot: %s）",
                prev_loss,
                cur_loss,
                abs(cur_loss - prev_loss),
                spike_dir,
                report_path,
                path,
            )

    state["prev_loss"] = cur_loss
    state["prev_step"] = optimization_step
    state["prev_snap"] = cur_entry


def _build_cosmos_demo_l1_log_dict(output_batch: dict, loss: torch.Tensor) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in _COSMOS_DEMO_L1_METRIC_KEYS:
        v = output_batch.get(k)
        out[k] = _tensor_to_log_float(v) if isinstance(v, torch.Tensor) else float("nan")
    out["cosmos_training_loss"] = _tensor_to_log_float(loss)
    return out


def _maybe_log_cosmos_demo_l1_metrics(
    *,
    output_batch: dict,
    loss: torch.Tensor,
    optimization_step: int,
    cosmos_train_epoch: int,
    wandb_logger: WandBLogger | None,
    cfg: TrainRLServerPipelineConfig,
    cosmos_cfg: any,
) -> None:
    """将五个 demo L1 与总训练 loss 按 ``cosmos_cfg.trainer.logging_iter`` 频率记录到 CSV 与 W&B（rank0）。"""
    if not imaginaire_distributed.is_rank0():
        return
    # logging_iter = int(getattr(cosmos_cfg.trainer, "logging_iter", 5) or 1)
    logging_iter = 30
    if optimization_step % logging_iter != 0:
        return

    metrics = _build_cosmos_demo_l1_log_dict(output_batch, loss)
    log_dir = os.path.join(cfg.output_dir, "logs_new")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "cosmos_demo_l1_metrics.csv")
    fieldnames = ["optimization_step", "epoch", *_COSMOS_DEMO_L1_METRIC_KEYS, "cosmos_training_loss"]
    row = {
        "optimization_step": optimization_step,
        "epoch": cosmos_train_epoch,
        **{k: metrics[k] for k in _COSMOS_DEMO_L1_METRIC_KEYS},
        "cosmos_training_loss": metrics["cosmos_training_loss"],
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

    if wandb_logger is not None:
        wandb_payload = {**metrics, "Optimization step": optimization_step}
        wandb_logger.log_dict(
            d=wandb_payload, mode="train", custom_step_key="Optimization step"
        )


def _ensure_cosmos_distributed_for_training(cosmos_cfg: any) -> None:
    """与 ImaginaireTrainer 一致：在实例化 cosmos 模型（FSDP mesh）前初始化分布式与 Megatron。

    ``Text2WorldModel`` 构造时会调用 ``hsdp_device_mesh()`` → ``imaginaire_distributed.get_world_size()``。
    若尚未 ``init_process_group``，该函数恒返回 1，会按 (1,1) 建 ``DeviceMesh``（仅含全局 rank 0）。
    ``torchrun`` 多进程下其它 rank 的 ``_dim_group_infos`` 为空，
    ``fully_shard`` 里 ``HSDPMeshInfo`` 访问 ``mesh.get_group(0)`` 会触发 ``IndexError``。

    多卡时还需 ``parallel_state.initialize_model_parallel``，否则 ``initialize_dataloader_train`` 里
    ``DistributedSampler`` 使用的 ``get_data_parallel_world_size()`` 等未就绪。
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    if not torch.distributed.is_initialized():
        with distributed_init():
            imaginaire_distributed.init()

    if getattr(cosmos_cfg, "model_parallel", None) is None:
        return
    if parallel_state.model_parallel_is_initialized():
        return

    mp = cosmos_cfg.model_parallel
    if (
        "create_gloo_process_groups"
        in inspect.signature(parallel_state.initialize_model_parallel).parameters
    ):
        parallel_state.initialize_model_parallel(
            pipeline_model_parallel_size=mp.pipeline_model_parallel_size,
            tensor_model_parallel_size=mp.tensor_model_parallel_size,
            context_parallel_size=mp.context_parallel_size,
            create_gloo_process_groups=False,
        )
    else:
        parallel_state.initialize_model_parallel(
            pipeline_model_parallel_size=mp.pipeline_model_parallel_size,
            tensor_model_parallel_size=mp.tensor_model_parallel_size,
            context_parallel_size=mp.context_parallel_size,
        )
    parallel_state.sequence_parallel = mp.sequence_parallel
    if parallel_state.sequence_parallel:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"


def initialize_cosmos_model(cosmos_cfg: any):
    cosmos_model = instantiate(cosmos_cfg.model)
    cosmos_model = cosmos_model.to("cuda", memory_format=cosmos_cfg.trainer.memory_format)  # type: ignore
    cosmos_model.on_train_start(cosmos_cfg.trainer.memory_format)
    return cosmos_model


def initialize_dataloader_train(cosmos_cfg: any):
    """
    初始化训练数据加载器和分布式采样器
    """
    # # cosmos5：初始化训练器,trainer.train已被拆分
    # trainer = cosmos_cfg.trainer.type(cosmos_cfg)
    # cosmos6：初始化数据加载器
    # 单机：DataLoader(dataset=..., shuffle=True)，不要 sampler。
    # 多卡：sampler=DistributedSampler(..., shuffle=False)，每个 epoch 调用 sampler.set_epoch(epoch)。
    dataset = instantiate(cosmos_cfg.dataloader_train.dataset)
    # 多卡使用分布式采样器
    sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=parallel_state.get_data_parallel_world_size(),
            rank=parallel_state.get_data_parallel_rank(),
            shuffle=True,  # 每个epoch先把「样本下标」打乱再分给各张卡，配合set_epoch(epoch)
            seed=0,
        )
    dataloader_train = DataLoader(
        dataset=dataset,
        # shuffle=True,  # 不使用分布式采样器，每个 epoch 调用 sampler.set_epoch(epoch)，确保每个 epoch 使用不同的随机种子
        sampler=sampler,
        batch_size=cosmos_cfg.dataloader_train.batch_size,
        drop_last=cosmos_cfg.dataloader_train.drop_last,
        num_workers=cosmos_cfg.dataloader_train.num_workers,
        persistent_workers=cosmos_cfg.dataloader_train.persistent_workers,
        pin_memory=cosmos_cfg.dataloader_train.pin_memory,
        pin_memory_device=cosmos_cfg.dataloader_train.pin_memory_device,
        timeout=cosmos_cfg.dataloader_train.timeout,
    )
    print("-----------------already create dataloader_train-----------------")
    # print('dataloader_train:', dataloader_train) # dataloader_train: <torch.utils.data.dataloader.DataLoader object at 0x7f504fe81a90>
    return dataloader_train
    # trainer.train(cosmos_model, dataloader_train, None)
    # print("-----------------already train cosmos model-----------------")
      

def get_dataloader_train_iter(cosmos_train_epoch: int, dataloader_train: DataLoader):
    # 按照epoch编号设置分布式采样器的随机顺序
    # dataloader_train.sampler.set_epoch(cosmos_train_epoch)
    dataloader_train_iter = iter(dataloader_train)
    return dataloader_train_iter


def get_forward_batch(cosmos_cfg: any, dataloader_train_iter: iter, straggler_detector: StragglerDetectorV2):
    with (
        straggler_detector.profile_section(  # 检测慢节点（数据加载部分）
                            "dataloading",
                            cosmos_cfg.trainer.straggler_detection.analyze_dataloading,
                            profile_cuda=False,
                        ),
    ):
        forward_batch = next(dataloader_train_iter)
    forward_batch = misc.to(forward_batch, device="cuda")

    # 对齐混合精度：padding_mask 若保持 float32，会在 DiT 内 concat_padding_mask 时
    # 把输入提升到 float32，从而与 bf16 权重产生 dtype mismatch（float != bfloat16），无法矩阵相乘
    if isinstance(forward_batch, dict) and "padding_mask" in forward_batch:
        if "video" in forward_batch and isinstance(forward_batch["video"], torch.Tensor):
            forward_batch["padding_mask"] = forward_batch["padding_mask"].type_as(forward_batch["video"])
        else:
            forward_batch["padding_mask"] = forward_batch["padding_mask"].to(dtype=torch.bfloat16)

    return forward_batch

def save_training_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    online_steps: int,
    interaction_message: dict | None,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer | None = None,
    dataset_repo_id: str | None = None,
    fps: int = 30,
) -> None:
    # 日志输出当前检查点保存的优化步数，便于调试和监控
    logging.info(f"Checkpoint policy after step {optimization_step}")
    
    # 计算步数显示的最小位数（确保目录命名对齐，如000001、000100）
    _num_digits = max(6, len(str(online_steps)))
    
    # 提取当前交互步数（若未收到交互信息则默认为0，用于恢复训练时对齐进度）
    interaction_step = interaction_message["Interaction step"] if interaction_message is not None else 0

    # 1. 创建检查点根目录：格式为 output_dir/checkpoints/step_xxx（xxx为总步数_当前步数）
    checkpoint_dir = get_step_checkpoint_dir(Path(cfg.output_dir), online_steps, optimization_step)
    
    # 2. 定义模型保存路径：检查点目录/pretrained_model/当前优化步数
    model_dir = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR, str(optimization_step))
    
    # 保存模型权重和配置
    # 会自动保存模型state_dict、配置文件等，支持后续from_pretrained加载
    policy.save_pretrained(model_dir)
    print('save model under', model_dir)

    # # （注释掉的备用逻辑）保存完整检查点（含优化器、调度器状态）
    # # 若需恢复训练时继续使用之前的优化器状态，需取消注释此段
    # save_checkpoint(
    #     checkpoint_dir=checkpoint_dir,
    #     step=optimization_step,
    #     cfg=cfg,
    #     policy=policy,
    #     optimizer=optimizers,
    #     scheduler=None,  # 本训练流程未使用学习率调度器，设为None
    # )

    # 3. 保存训练状态（优化步数+交互步数）
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)  # 确保目录存在，不存在则创建
    
    # 训练状态字典：包含恢复训练必需的核心进度信息
    training_state = {
        "step": optimization_step,  # 优化步数（恢复时从该步继续训练）
        "interaction_step": interaction_step  # 交互步数（对齐Actor端进度）
    }
    # 保存训练状态到文件
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))

    # 4. 更新"last"符号链接：指向当前最新检查点目录
    # 作用：快速访问最新模型，无需记住具体步数目录
    update_last_checkpoint(checkpoint_dir)

    # 5. 保存在线回放缓冲区为标准数据集（临时逻辑，后续可迁移到机器人端控制）
    # 数据集保存路径：output_dir/dataset
    dataset_dir = os.path.join(cfg.output_dir, "dataset")
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    # 确定数据集仓库ID：优先使用传入的dataset_repo_id，未指定则使用环境任务名
    repo_id_buffer_save = cfg.env.task if dataset_repo_id is None else dataset_repo_id
    
    # 将回放缓冲区转换为LeRobot标准数据集格式（支持后续加载复用）
    replay_buffer.to_lerobot_dataset(
        repo_id=repo_id_buffer_save,  # 数据集标识
        fps=fps,  # 与环境帧率一致，保证数据时间同步
        root=dataset_dir  # 保存根目录
    )

    # 6. 保存离线回放缓冲区为独立数据集
    if offline_replay_buffer is not None:
        # 离线数据集保存路径：output_dir/dataset_offline
        dataset_offline_dir = os.path.join(cfg.output_dir, "dataset_offline")
        
        # 若离线数据集目录已存在，先删除旧数据
        if os.path.exists(dataset_offline_dir) and os.path.isdir(dataset_offline_dir):
            shutil.rmtree(dataset_offline_dir)

        # 保存离线缓冲区为标准数据集（使用离线数据的repo_id标识）
        offline_replay_buffer.to_lerobot_dataset(
            cfg.dataset.repo_id,  # 离线数据集的仓库ID（从配置中读取）
            fps=fps,  # 保持与环境帧率一致
            root=dataset_offline_dir  # 离线数据集保存根目录
        )

    # 日志输出保存完成，提示支持恢复训练
    logging.info("Resume training")


def make_optimizers_and_scheduler(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    """
    Creates and returns optimizers for the actor, critic, and temperature components of a reinforcement learning policy.

    This function sets up Adam optimizers for:
    - The **actor network**, ensuring that only relevant parameters are optimized.
    - The **critic ensemble**, which evaluates the value function.
    - The **temperature parameter**, which controls the entropy in soft actor-critic (SAC)-like methods.

    It also initializes a learning rate scheduler, though currently, it is set to `None`.

    NOTE:
    - If the encoder is shared, its parameters are excluded from the actor's optimization process.
    - The policy's log temperature (`log_alpha`) is wrapped in a list to ensure proper optimization as a standalone tensor.

    Args:
        cfg: Configuration object containing hyperparameters.
        policy (nn.Module): The policy model containing the actor, critic, and temperature components.

    Returns:
        Tuple[Dict[str, torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
        A tuple containing:
        - `optimizers`: A dictionary mapping component names ("actor", "critic", "temperature") to their respective Adam optimizers.
        - `lr_scheduler`: Currently set to `None` but can be extended to support learning rate scheduling.

    """
    optimizer_critic = None
    optimizer_discrete_critic = None
    
    # 定义critic和discrete_critic优化器
    if "hgdagger" not in cfg.policy.type:
        optimizer_critic = torch.optim.Adam(params=list(policy.critic_ensemble.parameters()), lr=cfg.policy.critic_lr)
    
    actor_params = [
            p
            for n, p in policy.actor.named_parameters()
        ]

    if cfg.policy.num_discrete_actions is not None:
        if "silri" in cfg.policy.type or "hgdagger" in cfg.policy.type:
            actor_params = actor_params + list(policy.discrete_actor.parameters())
        else:
            optimizer_discrete_critic = torch.optim.Adam(
                params=policy.discrete_critic.parameters(), lr=cfg.policy.critic_lr
            )


    optimizer_actor = torch.optim.Adam(params=actor_params, lr=cfg.policy.actor_lr)


    lr_scheduler = None
    
    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
    }

    if "silri" in cfg.policy.type:
        optimizer_lagrange = torch.optim.Adam(params=list(policy.lagrange_net.parameters()), lr=0.01 * cfg.policy.critic_lr)
        optimizers["lagrange"] = optimizer_lagrange

        optimizer_expert = torch.optim.Adam(params=list(policy.expert_network.parameters()), lr=cfg.policy.actor_lr)
        optimizers["expert"] = optimizer_expert


    if "sac" in cfg.policy.type:
        optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=cfg.policy.critic_lr)
        optimizers["temperature"] = optimizer_temperature
    
    if optimizer_discrete_critic is not None:
        optimizers["discrete_critic"] = optimizer_discrete_critic
        
    return optimizers, lr_scheduler


def make_optimizers_and_scheduler_cosmos(cosmos_model: CosmosPolicyVideo2WorldModel, cosmos_cfg: any):
    """
    初始化优化器、学习率调度器、混合精度缩放器
    from /media/HIL-RL-Project/cosmos-policy/cosmos_policy/trainer.py def train
    """
    optimizer, scheduler = cosmos_model.init_optimizer_scheduler(cosmos_cfg.optimizer, cosmos_cfg.scheduler)
    # grad_scaler：只会在loss、backward之后调用，不会在forward之前调用
    grad_scaler = torch.amp.GradScaler("cuda", **cosmos_cfg.trainer.grad_scaler_args)
    return optimizer, scheduler, grad_scaler


def update_optimizer_cosmos(
    cosmos_cfg: any, 
    cosmos_model: CosmosPolicyVideo2WorldModel, 
    optimizer: torch.optim.Optimizer, 
    grad_scaler: torch.amp.GradScaler, 
    scheduler: torch.optim.lr_scheduler.LRScheduler, 
    grad_accum_iter: int, 
    straggler_detector: StragglerDetectorV2,
    optimization_step: int,
):
    # 梯度累加计数器递增
    grad_accum_iter += 1
    # 当累积步数达到配置要求时，执行优化器更新
    if grad_accum_iter == cosmos_cfg.trainer.grad_accum_iter:
        with straggler_detector.profile_section(  # 检测慢节点（优化器更新部分）
            "opt", cosmos_cfg.trainer.straggler_detection.analyze_optimizer,
            profile_cuda=False,
        ):
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        grad_accum_iter = 0

#################################################
# Training setup functions #
#################################################


def handle_resume_logic(cfg: TrainRLServerPipelineConfig) -> TrainRLServerPipelineConfig:
    """
    Handle the resume logic for training.

    If resume is True:
    - Verifies that a checkpoint exists
    - Loads the checkpoint configuration
    - Logs resumption details
    - Returns the checkpoint configuration

    If resume is False:
    - Checks if an output directory exists (to prevent accidental overwriting)
    - Returns the original configuration

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration

    Returns:
        TrainRLServerPipelineConfig: The updated configuration

    Raises:
        RuntimeError: If resume is True but no checkpoint found, or if resume is False but directory exists
    """
    out_dir = cfg.output_dir

    # Case 1: Not resuming, but need to check if directory exists to prevent overwrites
    if not cfg.resume:
        checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Output directory {checkpoint_dir} already exists. Use `resume=true` to resume training."
            )
        return cfg

    # Case 2: Resuming training
    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)

    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")

    # Log that we found a valid checkpoint and are resuming
    logging.info(
        colored(
            "Valid checkpoint found: resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"],
        )
    )

    # Load config using Draccus
    checkpoint_cfg_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR)

    checkpoint_cfg = TrainRLServerPipelineConfig.from_pretrained(checkpoint_cfg_path)

    # Ensure resume flag is set in returned config
    checkpoint_cfg.resume = True
    # todo: debug
    # checkpoint_cfg.output_dir = out_dir
    return checkpoint_cfg


def load_training_state(
    cfg: TrainRLServerPipelineConfig,
    optimizers: Optimizer | dict[str, Optimizer],
):
    """
    Loads the training state (optimizers, step count, etc.) from a checkpoint.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        optimizers (Optimizer | dict): Optimizers to load state into

    Returns:
        tuple: (optimization_step, interaction_step) or (None, None) if not resuming
    判断是否需要恢复训练，从保存的断点里加载优化器状态、加载训练步数（优化步数 + 交互步数）
    """
    if not cfg.resume:
        return None, None

    # Construct path to the last checkpoint directory
    checkpoint_dir = os.path.join(cfg.output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    print('------------------------------checkpoint_dir', checkpoint_dir)

    logging.info(f"Loading training state from {checkpoint_dir}")

    try:
        # Use the utility function from train_utils which loads the optimizer state
        step, optimizers, _ = utils_load_training_state(Path(checkpoint_dir), optimizers, None)

        # Load interaction step separately from training_state.pt
        training_state_path = os.path.join(checkpoint_dir, TRAINING_STATE_DIR, "training_state.pt")
        interaction_step = 0
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, weights_only=False)  # nosec B614: Safe usage of torch.load
            interaction_step = training_state.get("interaction_step", 0)

        logging.info(f"Resuming from step {step}, interaction step {interaction_step}")
        # step:模型更新步数；interaction_step：环境交互步数
        return step, interaction_step

    except Exception as e:
        logging.error(f"Failed to load training state: {e}")
        traceback.print_exc()
        exit(-1)
        return None, None


def load_training_state_cosmos(
    cosmos_cfg: any,
    cosmos_model: CosmosPolicyVideo2WorldModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
):
    """
    Loads the training state (optimizers, step count, etc.) from a checkpoint.

    ``checkpointer`` must be the same object used for saves (typically
    ``CosmosPolicyTrainer.checkpointer`` / ``ImaginaireTrainer.checkpointer``),
    not an attribute on ``cosmos_model``.
    """
    # checkpointer设置为全局变量
    global cosmos_checkpointer
    _cosmos_callbacks = imaginaire_callback.CallBackGroup(config=cosmos_cfg, trainer=None)
    
    cosmos_checkpointer = DistributedCheckpointer(
        cosmos_cfg.checkpoint,
        cosmos_cfg.job,
        callbacks=_cosmos_callbacks,
    )

    assert cosmos_checkpointer is not None

    iteration = cosmos_checkpointer.load(cosmos_model, optimizer, scheduler, grad_scaler)

    print("===========================================")
    print(os.path.dirname(os.path.abspath(__file__)))    # /media/HIL-RL-Project/HIL-RL
    print("===========================================")
    # grad_accum_iter：梯度累积步数
    # 设不设置为全局变量？，用于cosmos9中的梯度累积
    grad_accum_iter = 0
    return iteration, grad_accum_iter



def log_training_info(cfg: TrainRLServerPipelineConfig, policy: nn.Module) -> None:
    """
    Log information about the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        policy (nn.Module): Policy model
    """
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.policy.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(
    cfg: TrainRLServerPipelineConfig, device: str, storage_device: str, cosmos_cfg: any,
) -> ReplayBuffer:
    """
    Initialize a replay buffer, either empty or from a dataset if resuming.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized replay buffer
    """
    if not cfg.resume:
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )

    logging.info("Resume training load the online dataset")
    dataset_path = os.path.join(cfg.output_dir, "dataset")

    # NOTE: In RL is possible to not have a dataset.
    repo_id = None
    if cfg.dataset is not None:
        repo_id = cfg.dataset.repo_id
    # 实例化dataset
    if cfg.policy.type == "cosmos":
        dataset = instantiate(cosmos_cfg.dataloader_train.dataset)
    else:   
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_path,
        )

    return ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        capacity=cfg.policy.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        optimize_memory=True
    )

import traceback
import sys

def initialize_offline_replay_buffer(
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device: str,
    cosmos_cfg: any,
) -> ReplayBuffer:
    """
    Initialize an offline replay buffer from a dataset.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized offline replay buffer
    """
    if not cfg.resume:
        logging.info("make_dataset offline buffer")
        # INSERT_YOUR_CODE
        # 但是cosmos本身的实现没有buffer，如果用原来的libero的话，需要把buffer改动成dataloader，修改太多
        offline_dataset = make_dataset(cfg)
        # if cfg.policy.type == "cosmos":  # 暂时用libero的环境和数据集
        #     offline_dataset = instantiate(cosmos_cfg.dataloader_train.dataset)
        # else:
        #     offline_dataset = make_dataset(cfg)  # 实例化数据集LeRobotDataset，但是是否需要修改成

    else:
        logging.info("load offline dataset")
        dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=dataset_offline_path,
        )

        
    logging.info("Convert to a offline replay buffer")
    try:
        offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
            offline_dataset,
            device=device,
            state_keys=cfg.policy.input_features.keys(),   # cosmos的input_feature应该是什么样的？
            storage_device=storage_device,
            optimize_memory=True,
            capacity=cfg.policy.offline_buffer_capacity
        )
    except Exception as e:
        print(f"[{type(e).__name__}] {e!r}")
        traceback.print_exc()          # full stacktrace
        sys.exit(1)
    return offline_replay_buffer


#################################################
# Utilities/Helpers functions #
#################################################


def get_observation_features(
    policy, observations: torch.Tensor, next_observations: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Get observation features from the policy encoder. It act as cache for the observation features.
    when the encoder is frozen, the observation features are not updated.
    We can save compute by caching the observation features.

    Args:
        policy: The policy model
        observations: The current observations
        next_observations: The next observations

    Returns:
        tuple: observation_features, next_observation_features
    """

    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None

    with torch.no_grad():
        observation_features = policy.actor.encoder.get_cached_image_features(observations, normalize=True)
        next_observation_features = policy.actor.encoder.get_cached_image_features(
            next_observations, normalize=True
        )


    return observation_features, next_observation_features


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.learner == "threads"


def check_nan_in_transition(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_state: torch.Tensor,
    raise_error: bool = False,
) -> bool:
    """
    Check for NaN values in transition data.

    Args:
        observations: Dictionary of observation tensors
        actions: Action tensor
        next_state: Dictionary of next state tensors
        raise_error: If True, raises ValueError when NaN is detected

    Returns:
        bool: True if NaN values were detected, False otherwise
    """
    nan_detected = False

    # Check observations
    for key, tensor in observations.items():
        if torch.isnan(tensor).any():
            logging.error(f"observations[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in observations[{key}]")

    # Check next state
    for key, tensor in next_state.items():
        if torch.isnan(tensor).any():
            logging.error(f"next_state[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in next_state[{key}]")

    # Check actions
    if torch.isnan(actions).any():
        logging.error("actions contains NaN values")
        nan_detected = True
        if raise_error:
            raise ValueError("NaN detected in actions")

    return nan_detected


def push_actor_policy_to_queue(parameters_queue: Queue, policy: nn.Module):
    logging.debug("[LEARNER] Pushing actor policy to the queue")

    # Create a dictionary to hold all the state dicts
    state_dicts = {"policy": move_state_dict_to_device(policy.actor.state_dict(), device="cpu")}

    # Add discrete critic if it exists
    if hasattr(policy, "discrete_critic") and policy.discrete_critic is not None:
        state_dicts["discrete_critic"] = move_state_dict_to_device(
            policy.discrete_critic.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including discrete critic in state dict push")
    
    if hasattr(policy, "discrete_actor") and policy.discrete_actor is not None:
        state_dicts["discrete_actor"] = move_state_dict_to_device(
            policy.discrete_actor.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including discrete actor in state dict push")

    state_bytes = state_to_bytes(state_dicts)
    parameters_queue.put(state_bytes)


def process_interaction_message(
    message, interaction_step_shift: int, wandb_logger: WandBLogger | None = None
):
    """Process a single interaction message with consistent handling."""
    message = bytes_to_python_object(message)
    # Shift interaction step for consistency with checkpointed state
    message["Interaction step"] += interaction_step_shift

    # Log if logger available
    if wandb_logger:
        wandb_logger.log_dict(d=message, mode="train", custom_step_key="Interaction step")

    return message


def process_transitions(
    optimization_step: int,
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    device: str,
    dataset_repo_id: str | None,
    shutdown_event: any,
    optimizers: dict[str, Optimizer],
    policy: nn.Module,
    clip_grad_norm_value: float,
    batch_size: int,
    async_prefetch: bool,
    wandb_logger: WandBLogger | None,
    cfg: TrainRLServerPipelineConfig,
):
    """Process all available transitions from the queue.

    Args:
        transition_queue: Queue for receiving transitions from the actor
        replay_buffer: Replay buffer to add transitions to
        offline_replay_buffer: Offline replay buffer to add transitions to
        device: Device to move transitions to
        dataset_repo_id: Repository ID for dataset
        shutdown_event: Event to signal shutdown
    """
    global new_offline_transition_num
    while not transition_queue.empty() and not shutdown_event.is_set():
        transition_list = transition_queue.get()
        start_get_transition_time = time.time()
        transition_list = bytes_to_transitions(buffer=transition_list)
        end_get_transition_time = time.time()

        global transition_cnt
        transition_cnt += 1
        global transition_time
        transition_time += end_get_transition_time - start_get_transition_time
        # print(f"==========================================get transition time: {transition_time / transition_cnt:.6f}s {transition_cnt} transitions")

        new_transition_num = 0
        success_list = []
        reward_list = []


        for transition in transition_list:
            transition['complementary_info']['target_prob'] = 0
            transition = move_transition_to_device(transition=transition, device=device)

            # Skip transitions with NaN values
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition["action"],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            # Add all valid data to the main online buffer
            replay_buffer.add(**transition)
            # Add data with intervention to the offline buffer
            if dataset_repo_id is not None and transition.get("complementary_info", {}).get(
                "is_intervention"
            ):
                offline_replay_buffer.add(**transition)
                new_offline_transition_num += 1
                if new_offline_transition_num % 50 == 0 and "silri" in cfg.policy.type:
                    expert_training(offline_replay_buffer, optimizers, policy, clip_grad_norm_value, device, async_prefetch, wandb_logger, optimization_step)
                
            new_transition_num += 1


def process_interaction_messages(
    interaction_message_queue: Queue,
    interaction_step_shift: int,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,
) -> dict | None:
    """Process all available interaction messages from the queue.

    Args:
        interaction_message_queue: Queue for receiving interaction messages
        interaction_step_shift: Amount to shift interaction step by
        wandb_logger: Logger for tracking progress
        shutdown_event: Event to signal shutdown

    Returns:
        dict | None: The last interaction message processed, or None if none were processed
    """
    last_message = None
    while not interaction_message_queue.empty() and not shutdown_event.is_set():
        message = interaction_message_queue.get()
        last_message = process_interaction_message(
            message=message,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
        )

    return last_message


if __name__ == "__main__":
    try:
        train_cli()
        logging.info("[LEARNER] main finished")
    except Exception as e:
        print(f"[{type(e).__name__}] {e!r}")
        traceback.print_exc()          # full stacktrace
        sys.exit(1)