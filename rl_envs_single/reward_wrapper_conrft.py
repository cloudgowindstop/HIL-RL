import jax
import jax.numpy as jnp
from typing import Callable, Dict, List
from flax.training import checkpoints
import flax.linen as nn
import optax 
import pickle as pkl
from flax.training.train_state import TrainState 
from rl_envs.utils.encoding import EncodingWrapper
from rl_envs.utils.resnet_v1 import resnetv1_configs, PreTrainedResNetEncoder
from rl_envs.utils.replay_buffer import ReplayBuffer
from rl_envs.shared_state import shared_state
from rl_envs.utils.train_utils import concat_batches
from rl_envs.utils.data_augmentations import data_augmentation_fn
# from rl_envs.utils.reward_classifier import train_step
import json 

@jax.jit
def train_step(state, batch, key):
    def loss_fn(params):
        # print('inputs:', batch["observations"]['right'].shape, batch["observations"]["state"].shape)
        logits = state.apply_fn({"params": params}, batch["observations"], rngs={"dropout": key}, train=True)
        # print('logits:', logits)
        # print(">>>>>>>>>>>>  in compute loss logits shape: ", logits.shape, "labels shape: ", batch["labels"].shape)
        return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    # print("right shape: ", batch["observations"]["right"].shape)
    logits = state.apply_fn({"params": state.params}, batch["observations"], train=False, rngs={"dropout": key})
    train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

    # print("logits shape: ", logits.shape)

    return state.apply_gradients(grads=grads), loss, train_accuracy, nn.sigmoid(logits)


class BinaryClassifier(nn.Module):
    encoder_def: nn.Module
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class NWayClassifier(nn.Module):
    encoder_def: nn.Module
    hidden_dim: int = 256
    n_way: int = 3

    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_way)(x)
        return x




def create_classifier(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    pretrained_encoder_path: str=None,
    enable_stacking: bool = True,
    n_way: int = 2,
):
    pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
        pre_pooling=True,
        name="pretrained_encoder",
    )
    encoders = {
        image_key: PreTrainedResNetEncoder(
            pooling_method="spatial_learned_embeddings",
            num_spatial_blocks=8,
            bottleneck_dim=256,
            pretrained_encoder=pretrained_encoder,
            name=f"encoder_{image_key}",
        )
        for image_key in image_keys
    }
    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=False,
        enable_stacking=enable_stacking,
        image_keys=image_keys,
    )
    if n_way == 2:
        classifier_def = BinaryClassifier(encoder_def=encoder_def)
    else:
        classifier_def = NWayClassifier(encoder_def=encoder_def, n_way=n_way)
    params = classifier_def.init(key, sample)["params"]
    classifier = TrainState.create(
        apply_fn=classifier_def.apply,
        params=params,
        tx=optax.adam(learning_rate=1e-4),
    )
   
    if pretrained_encoder_path is not None:
        
        import os

        # 方法1: 使用 os.getcwd() (最常用)
        current_dir = os.getcwd()
        print(f"当前工作目录: {current_dir}")

        # 方法2: 使用 pathlib (Python 3.4+)
        from pathlib import Path
        current_dir = Path.cwd()
        print(f"当前工作目录: {current_dir}")

        # 方法3: 使用 os.path
        import os.path
        current_dir = os.path.abspath('.')
        print(f"当前工作目录: {current_dir}")


        with open(pretrained_encoder_path, "rb") as f:
            encoder_params = pkl.load(f)
        param_count = sum(x.size for x in jax.tree_leaves(encoder_params))
        print(
            f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
        )
        new_params = classifier.params
        for image_key in image_keys:
            if "pretrained_encoder" in new_params["encoder_def"][f"encoder_{image_key}"]:
                for k in new_params["encoder_def"][f"encoder_{image_key}"][
                    "pretrained_encoder"
                ]:
                    if k in encoder_params:
                        new_params["encoder_def"][f"encoder_{image_key}"][
                            "pretrained_encoder"
                        ][k] = encoder_params[k]
                        print(f"replaced {k} in encoder_{image_key}")

        classifier = classifier.replace(params=new_params)
    return classifier


def load_classifier(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    checkpoint_path: str,
    n_way: int = 2,
    pretrained_encoder_path: str = None,
    enable_stacking: bool = True,
) -> Callable[[Dict], jnp.ndarray]:
    """
    Return: a function that takes in an observation
            and returns the logits of the classifier.
    """
    classifier = create_classifier(key, sample, image_keys, n_way=n_way, pretrained_encoder_path=pretrained_encoder_path, enable_stacking=enable_stacking)
    classifier = checkpoints.restore_checkpoint(
        checkpoint_path,
        target=classifier,
    )
    
    return classifier

import time
import gymnasium as gym
import os
import glob
import cv2

import copy
def transform_data(data,algorithm="hil-serl"):
    for obs_key in ["observations", "next_observations"]:
        for key in data[obs_key].keys():
            if len(data[obs_key][key].shape) ==1:
                continue
            elif len(data[obs_key][key].shape) == 2:
                data[obs_key][key] = np.array([data[obs_key][key][-1]])
            elif len(data[obs_key][key].shape) == 3:
                if algorithm == "hil-serl":
                    data[obs_key][key] = np.array([cv2.resize(data[obs_key][key], (128, 128))])
                elif algorithm == "conrft":
                    if key == "right":
                        data[obs_key][key] = cv2.resize(data[obs_key][key], (256, 256))
                    elif key == "wrist":
                        data[obs_key][key] = cv2.resize(data[obs_key][key], (128, 128))
            elif len(data[obs_key][key].shape) == 4:
                if algorithm == "hil-serl":
                    data[obs_key][key] = np.array([cv2.resize(data[obs_key][key][-1], (128, 128))])
                elif algorithm == "conrft":
                    if key == "right":
                        data[obs_key][key] = np.array([cv2.resize(data[obs_key][key][-1], (256, 256))])
                    elif key == "wrist":
                        data[obs_key][key] = cv2.resize(data[obs_key][key][-1], (128, 128))
            else:
                raise ValueError(f"Unsupported shape: {data[obs_key][key].shape}")
            
    return data
            
import datetime
import numpy as np


def _resize_hwc_to(img, size_hw):
    """Resize images for the reward classifier. Supports (H,W,C) -> (1,H',W',C) or
    (..., H,W,C) -> (..., H',W',C) with per-frame cv2.resize (leading dims preserved)."""
    img = np.asarray(img)
    if img.ndim == 3:
        return np.array([cv2.resize(img, size_hw)])
    if img.ndim >= 4:
        leading = img.shape[:-3]
        h, w, c = img.shape[-3:]
        flat = img.reshape(-1, h, w, c)
        out = np.stack([cv2.resize(flat[i], size_hw) for i in range(flat.shape[0])], axis=0)
        return out.reshape(*leading, out.shape[-3], out.shape[-2], out.shape[-1])
    raise ValueError(f"Expected image with at least 3 dims, got shape {img.shape}")


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env, classifier_cfg, use_force=False):
        super().__init__(env)
        # self.reward_classifier_func = reward_classifier_func
        self.target_hz = classifier_cfg.target_hz
        observation_space = copy.deepcopy(env.observation_space)
        self.algorithm = classifier_cfg.algorithm
        
        self.classifier = load_classifier(
            key=jax.random.PRNGKey(0),
            sample=observation_space.sample(),
            image_keys=classifier_cfg.classifier_keys,
            checkpoint_path=classifier_cfg.checkpoint_path,
            pretrained_encoder_path=classifier_cfg.pretrained_encoder_path,
            enable_stacking=classifier_cfg.enable_stacking,
        )
        self.time_step = 0
        self.new_data_count = 0
        self.reward_pos = classifier_cfg.reward_pos
        self.reward_neg = classifier_cfg.reward_neg
        self.classifier_keys = classifier_cfg.classifier_keys
        self.task_name = classifier_cfg.task_name
        self.train_epoch = 0
        self.batch_size = classifier_cfg.batch_size
        self.enable_stacking = classifier_cfg.enable_stacking
        print('------> classifier_cfg.checkpoint_path:', classifier_cfg.checkpoint_path)
        for _, all_dirs, _ in os.walk(classifier_cfg.checkpoint_path):
            for model_path in all_dirs:
                if model_path.split('_')[0] == 'checkpoint':
                    train_epoch = int(model_path.split('_')[-1])
                    self.train_epoch = max(self.train_epoch, train_epoch)
                    break 
        assert self.train_epoch > 0, 'train_epoch is 0'

        self.save_dir = os.path.join(os.getcwd(),classifier_cfg.checkpoint_path)
        print('------> save model to :', self.save_dir)
        self.require_train = classifier_cfg.require_train
        if self.require_train:
            devices = jax.local_devices()
            self.sharding = jax.sharding.PositionalSharding(devices)
            self.rng = jax.random.PRNGKey(0)
            classifier_root_dir = os.path.dirname(classifier_cfg.checkpoint_path)
            online_data_dir = os.path.join(classifier_root_dir, "classifier_data_online")
            if not os.path.exists(online_data_dir):
                os.makedirs(online_data_dir)
            self.classifier_root_dir = classifier_root_dir
            self.pos_buffer = ReplayBuffer(observation_space, env.action_space, capacity=20000, include_label=True,)
            self.neg_buffer = ReplayBuffer(observation_space, env.action_space, capacity=50000, include_label=True,)
            print('classifier_root_dir:', classifier_root_dir)

            """
            读取全部的pkl文件，根据reward区分success和failure，insert到不同的buffer中
            """
            classifier_root_dir = os.path.dirname(classifier_cfg.checkpoint_path)
            file_path = os.path.join(classifier_root_dir, "classifier_data")
            
            pkl_files = []
            
            for root, dirs, files in os.walk(file_path):
                for file in files:
                    if file.endswith('.pkl'):
                        pkl_files.append(os.path.join(root, file))

            for path in pkl_files:
                raw_data = pkl.load(open(path, "rb"))
                if self.algorithm == "conrft":
                    right_image = raw_data["images"]["right"]
                    wrist_image = raw_data["images"]["wrist"]
                else:
                    raw_data["images"]["right"] = cv2.imdecode(raw_data["images"]["right"], cv2.IMREAD_COLOR)
                    raw_data["images"]["wrist"] = cv2.imdecode(raw_data["images"]["wrist"], cv2.IMREAD_COLOR)
                    raw_data["images"]["right"] = cv2.cvtColor(raw_data["images"]["right"], cv2.COLOR_RGB2BGR)
                    raw_data["images"]["wrist"] = cv2.cvtColor(raw_data["images"]["wrist"], cv2.COLOR_RGB2BGR)
                    right_image = np.array(cv2.resize(raw_data["images"]["right"][0:480,225:-50,:], (128, 128)))
                    wrist_image = np.array(cv2.resize(raw_data["images"]["wrist"], (128, 128)))

                # if not os.path.exists(f"{self.save_dir}/offline_right_image.png"):
                #     cv2.imwrite(f"{self.save_dir}/offline_right_image.png", right_image)
                #     cv2.imwrite(f"{self.save_dir}/offline_wrist_image.png", wrist_image)
                #     print("offline_right_image shape:", right_image.shape)
                #     print("offline_wrist_image shape:", wrist_image.shape)
                #     print('offline_right_image has been saved')
                if self.algorithm == "conrft":
                    state = raw_data["observations"]["state"]
                else:
                    if use_force:
                        state = np.concatenate([raw_data["arm_pose"]["single"],[raw_data["hand_joints"]["single"]],raw_data["ee_force"]["single"]],axis=-1)
                    else:
                        state = np.concatenate([raw_data["arm_pose"]["single"],raw_data["hand_joints"]["single"]],axis=-1)
                if raw_data["reward"] > 0:
                    label = 1
                else:
                    label = 0
                
                
                new_data ={
                        "observations":{
                                        "right": right_image,
                                        "wrist": wrist_image,
                                        "state": state,
                        },
                        "next_observations":{
                                        "right": right_image,
                                        "wrist": wrist_image,
                                        "state": state,
                        },
                        "labels": label,
                        "actions": env.action_space.sample(),
                        "masks": 1,
                        "dones": 0,
                        "rewards": raw_data["reward"],
                }

                if label == 1:
                    self.pos_buffer.insert(new_data)
                elif label == 0:
                    self.neg_buffer.insert(new_data)
                uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            
            # success_paths = glob.glob(os.path.join(classifier_root_dir, "classifier_data", "*success*.pkl"))
            # print(f"success_paths: {len(success_paths)}")
            # input("Press Enter to continue...")
            # for path in success_paths:
            #     success_data = pkl.load(open(path, "rb"))
            #     success_data['labels'] = 1
            #     success_data['actions'] = env.action_space.sample()
            #     success_data = transform_data(success_data,algorithm=classifier_cfg.algorithm)
            #     uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            #     # rgb_image = success_data['observations']['right']
            #     # print(f"success_data['observations']['right'].shape: {rgb_image.shape}")
            #     # cv2.imwrite(f"{self.save_dir}/train_success_{uuid}.png", rgb_image[0])
            #     # print(f"saved success image to {self.save_dir}/train_success_{uuid}.png")
            #     self.pos_buffer.insert(success_data)

            # failure_paths = glob.glob(os.path.join(classifier_root_dir, "classifier_data", "*failure*.pkl"))
            # for path in failure_paths:
            #     failure_data = pkl.load(open(path, "rb"))
            #     failure_data['labels'] = 0
            #     failure_data['actions'] = env.action_space.sample()
            #     failure_data = transform_data(failure_data,algorithm=classifier_cfg.algorithm)
            #     # uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            #     # rgb_image = failure_data['observations']['right']
            #     # print(f"failure_data['observations']['right'].shape: {rgb_image.shape}")
            #     # cv2.imwrite(f"{self.save_dir}/train_failure_{uuid}.png", rgb_image[0])
            #     self.neg_buffer.insert(failure_data)


    def compute_reward(self, obs): 
        result = self.classifier.apply_fn({"params": self.classifier.params}, obs, train=False)[0] 
        result = 1 / (1 + jnp.exp(-result))
        # if sigmoid(result) > 0.9 and env.curr_gripper_pos > 0.5 and env.currpos[2] > 0.16:
        # print('classifier result:', result)
        if result > 0.9 :
            if self.task_name == 'fold_rag' or self.task_name == 'pick_place_bread': 
                if self.env.unwrapped.curr_gripper_joints < 0.5:
                    return self.reward_pos
                else:
                    return self.reward_neg
            return self.reward_pos
        else:
            return self.reward_neg

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = copy.deepcopy(obs)
        if self.algorithm == "hil-serl":
            for key in obs.keys():
                # self.last_obs[key] = self.last_obs[key][np.newaxis, :]
                self.last_obs[key] = self.last_obs[key]
        self.time_step = 0
        info['succeed'] = False
        return obs, info
    
    def step(self, action):
        step_st_time = time.time()
        self.time_step += 1
        obs, rew, _, truncated, info = self.env.step(action)
        reward_st_time = time.time()
        reward_obs = copy.deepcopy(obs)
        reward_obs['right'] = _resize_hwc_to(reward_obs['right'], (128, 128))
        reward_obs['wrist'] = _resize_hwc_to(reward_obs['wrist'], (128, 128))
        rew = self.compute_reward(reward_obs)
        terminated = rew > 0.5 
        if truncated:   
            print("任务超时，将自动重置")
        classifier_need_update = False
        
        if self.time_step <= 10 or truncated:
            terminated = False
            shared_state.terminate = False
            rew = self.reward_neg
        else:
            if terminated:   
                start_time = time.time()
                # shared_state.terminate = True
                print("模型判别任务成功结束，判断错误请按 space 键更正, 否则将在5秒后自动继续")
                while True:
                    if shared_state.terminate:
                        rew = self.reward_neg
                        terminated = False
                        classifier_need_update = True
                        shared_state.terminate = False
                        break
                    if time.time() - start_time > 5:
                        break
            elif shared_state.terminate:
                classifier_need_update = True
                terminated = True
                rew = self.reward_pos
                shared_state.terminate = False
        
        if classifier_need_update and self.require_train :
            self.new_data_count += 1
            # print('add one transition into buffer and train...')

            transition = dict(
                observations=self.last_obs,
                actions=action,
                next_observations=self.last_obs,
                rewards=rew,
                masks=1.0-terminated,
                dones=terminated,
            )

            # if not os.path.exists(f"{self.save_dir}/online_right_image.png"):
            #     cv2.imwrite(f"{self.save_dir}/online_right_image.png", self.last_obs["right"])
            #     cv2.imwrite(f"{self.save_dir}/online_wrist_image.png", self.last_obs["wrist"])
            #     print("online_right_image shape:", self.last_obs["right"].shape)
            #     print("online_wrist_image shape:", self.last_obs["wrist"].shape)
            #     print('online_right_image has been saved')

            uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


            if rew > 0.5: 
                transition['labels'] = 1
                # print('transition[actions]: ', transition['actions'])
                self.pos_buffer.insert(transition)
                if self.require_train:
                    file_name = os.path.join(self.classifier_root_dir, "classifier_data_online", f"success_{uuid}.pkl")
                    with open(file_name, "wb") as f:
                        pkl.dump(transition, f)
                        print(f"saved successful transitions to {file_name}")
            else:
                transition['labels'] = 0
                # print('transition[actions]: ', transition['actions'])
                self.neg_buffer.insert(transition)
                if self.require_train:
                    file_name = os.path.join(self.classifier_root_dir, "classifier_data_online", f"failure_{uuid}.pkl")
                    with open(file_name, "wb") as f:
                        pkl.dump(transition, f)
                        print(f"saved failure transitions to {file_name}")
        
            if self.require_train:
                
                pos_iterator = self.pos_buffer.get_iterator(sample_args={"batch_size": self.batch_size,}, device=self.sharding.replicate(),)
                neg_iterator = self.neg_buffer.get_iterator(sample_args={"batch_size": self.batch_size,}, device=self.sharding.replicate(),)


                # Sample equal number of positive and negative examples
                pos_sample = next(pos_iterator)
                neg_sample = next(neg_iterator)
                # Merge and create labels
                batch = concat_batches(pos_sample, neg_sample, axis=0)

                obs_for_classifier = batch["observations"]
                img_updates = {}
                for k in self.classifier_keys:
                    if k in obs_for_classifier:
                        img_updates[k] = jnp.asarray(
                            _resize_hwc_to(jax.device_get(obs_for_classifier[k]), (128, 128))
                        )
                obs_for_classifier = obs_for_classifier.copy(add_or_replace=img_updates)
                batch = batch.copy(add_or_replace={"observations": obs_for_classifier})

                rng, key = jax.random.split(self.rng)
                if self.enable_stacking:
                    num_batch_dims = 2
                else:
                    num_batch_dims = 1
                obs_classifier = data_augmentation_fn(key, batch["observations"], self.classifier_keys, num_batch_dims=num_batch_dims) 
                batch = batch.copy(
                    add_or_replace={
                        "observations": obs_classifier,
                        "labels": batch["labels"][..., None],
                    }
                )

                # if len(batch['observations']['state'].shape) == 2:
                #     batch['observations']['state'] = batch['observations']['state'][:, np.newaxis, :]
                # if len(batch['observations']['right'].shape) == 2:
                #     batch['observations']['right'] = batch['observations']['right'][:, np.newaxis, :]
                    
                rng, key = jax.random.split(self.rng)
                for i in range(10):
                    self.classifier, train_loss, train_accuracy, output_results = train_step(self.classifier, batch, key)

                    log_info = {
                        "epoch": self.train_epoch,
                        "train_loss": float(train_loss),
                        "train_accuracy": float(train_accuracy),
                    }
                    # print('------ info -------:', info)
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
                    with open(os.path.join(self.save_dir, "classifier_log.jsonl"), "a") as f:
                        f.write(json.dumps(log_info, default=str) + "\n")
                    
                    # labels_host = np.asarray(jax.device_get(batch["labels"]).squeeze(-1))  # [B]
                    # right_imgs  = np.asarray(jax.device_get(batch["observations"]["right"]))  # [B, H, W, C]

                    # results = np.asarray(jax.device_get(output_results))
                    # # print('------ results -------:', results)

                    # # 取成功/失败的下标
                    # success_idx = np.where(labels_host == 1)[0][:10]
                    # failure_idx = np.where(labels_host == 0)[0][:10]

                    # print("right keys:", list(batch["observations"].keys()))
                    # print("right shape:", right_imgs.shape)
                    # print(f"success_count: {success_idx.size}, failure_count: {failure_idx.size}")

                    # 保存样本
                    # for idx in success_idx:
                    #     cv2.imwrite(f"{self.save_dir}/train_success_{idx}_{results[idx][0]:.2f}.png", right_imgs[idx])
                    # for idx in failure_idx:
                    #     cv2.imwrite(f"{self.save_dir}/train_failure_{idx}_{results[idx][0]:.2f}.png", right_imgs[idx])
                self.train_epoch += 1
                
                if self.train_epoch % 10 == 0:
                    checkpoints.save_checkpoint(self.save_dir, self.classifier, step=self.train_epoch, overwrite=True)
                


        self.last_obs = copy.deepcopy(obs)
        if self.algorithm == "hil-serl":
            for key in obs.keys():
                # self.last_obs[key] = self.last_obs[key][np.newaxis, :]
                self.last_obs[key] = self.last_obs[key]
        info['succeed'] = bool(terminated)
        # step_end_time = time.time()
        # if self.target_hz is not None:
        #     time.sleep(max(0, 1/self.target_hz - (step_end_time - step_st_time)))

        return obs, rew, terminated, truncated, info


class HumanInterventionRewardWrapper(gym.Wrapper):
    def __init__(self, env, intervene_penalty=-0.02):
        super().__init__(env)
        self.intervene_penalty = intervene_penalty
        self.last_obs = None
        self.already_intervene = False
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.already_intervene = False
        return obs, info
    
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            if not self.already_intervene:
                rew += self.intervene_penalty
                self.already_intervene = True
        else:
            self.already_intervene = False

        return obs, rew, terminated, truncated, info


class VelocityPenaltyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.velocity_penalty = -1.0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        pre_pose = self.env.unwrapped.currpos
        obs, rew, terminated, truncated, info = self.env.step(action)
        cur_pose = self.env.unwrapped.currpos
        pose_diff = np.linalg.norm(cur_pose - pre_pose, ord=1)
        action_diff = np.linalg.norm(action, ord=1)
        velocity_penalty = (pose_diff + action_diff) * self.velocity_penalty
        rew += velocity_penalty
        rew = float(rew)
        return obs, rew, terminated, truncated, info
    


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = self.env.unwrapped.curr_gripper_joints
        info['discrete_penalty'] = 0.0
        info['grasp_penalty'] = 0.0
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        action = copy.deepcopy(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]


        if (action[-1] > 0.3 and self.last_gripper_pos < 0.3) or (
            action[-1] < 0.5 and self.last_gripper_pos > 0.5):
            # info['grasp_penalty'] = -0.2
            # info['discrete_penalty'] = -0.2
            info['grasp_penalty'] = -0.5
            info['discrete_penalty'] = -0.5
        else:
            info['grasp_penalty'] = 0.0
            info['discrete_penalty'] = 0.0
        self.last_gripper_pos = self.env.unwrapped.curr_gripper_joints
        
        return observation, reward, terminated, truncated, info
    