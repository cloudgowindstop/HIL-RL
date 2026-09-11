# 临时禁用conda path
source /media/HIL-RL-Project/HIL-RL/.venv/bin/activate
export PATH=/media/HIL-RL-Project/HIL-RL/.venv/bin:$PATH
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# 1. 设置 WandB API 密钥（确保生效）
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in the environment}"
# 2. 切换回 legacy 后端（规避 wandb-core 兼容性问题，可选但推荐）
export WANDB__REQUIRE_LEGACY_SERVICE=TRUE
echo "已启用 WandB legacy 后端，避免 wandb-core 超时"
# 3. 延长初始化超时（通过环境变量传递，无需改代码，优先级低于脚本内设置）
export WANDB_INIT_TIMEOUT=180  # 超时时间180秒，可根据网络调整
echo "WandB 初始化超时已设置为 $WANDB_INIT_TIMEOUT 秒"

# 清理 :99 残留锁文件，重启 Xvfb
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
sleep 1
export DISPLAY=:99
# MuJoCo 使用 OSMesa 软件渲染
unset http_proxy
unset https_proxy
apt-get update
apt-get install -y \
    libosmesa6 \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
hash -r
# 绕过tokenizers path
export PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\n' | grep -v cosmos_policy | tr '\n' ':')

# 导入libero的config路径
export LIBERO_CONFIG_PATH=/media/HIL-RL-Project/cosmos-policy/cosmos_policy/config/libero


# 指定huggingface保存的路径
export HF_HUB_CACHE=/media/cosmos-policy/cosmos_policy/models/Cosmos-Policy-LIBERO-Predict2-2B/

export PYTHONPATH=$PYTHONPATH:../../../cosmos-policy/
export PYTHONPATH=$PYTHONPATH:../../../cosmos-policy/cosmos_policy/
export PYTHONPATH=$PYTHONPATH:/media/HIL-RL-Project/HIL-RL/lerobot/src/
# export PYTHONPATH=$PYTHONPATH:../../../lerobot_new/src/
# export PYTHONPATH=$PYTHONPATH:../../../RL-Robot-Env/
export PYTHONPATH=$PYTHONPATH:../../rl_envs/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL
export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
# export http_proxy=http://127.0.0.1:7890 && export https_proxy=http://127.0.0.1:7890

# export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000

# python3 -m pip install protobuf==6.32.0 \
#     -i https://pypi.org/simple

# Hugging Face Hub mirror (no proxy). Common choice in CN: https://hf-mirror.com
# You can override per-run: HF_ENDPOINT=... ./learner.sh

# task_name=close_trashbin_franka_1028
task_name=pick_toy
mkdir -p experiments/${task_name}
cd experiments/${task_name}

# 指定保存权重路径
export IMAGINAIRE_OUTPUT_ROOT=/media/HIL-RL-Project/HIL-RL/experiments/${task_name}

# 多卡 DDP 启动（在此直接改可见卡 / NPROC / MASTER_PORT）
export CUDA_VISIBLE_DEVICES=0
NPROC=1
MASTER_PORT=29511

# VAE encode 用哪些卡（进程内 cuda 下标，逗号分隔）
# 例：可见 0,1,2,3 时，ENCODE_CUDA_DEVICES=1,2,3 表示用其中第 2/3/4 张做 encode
# 不设则按 ENCODE_WORLD_SIZE 使用 cuda:0 .. N-1
export ENCODE_CUDA_DEVICES=0
# export ENCODE_WORLD_SIZE=4   # 仅在未设置 ENCODE_CUDA_DEVICES 时生效

torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT} ../../learner_copy_dist.py \
        robot_type@_global_=franka \
        task@_global_=${task_name} \
        policy_type=cosmos
