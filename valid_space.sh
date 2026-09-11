# 临时禁用conda path
source /home/ubuntu/Dev/hermine/HIL-RL-Project/.venv/bin/activate
export PATH=/home/ubuntu/Dev/hermine/HIL-RL-Project/.venv/bin:$PATH
# # 清理 :99 残留锁文件，重启 Xvfb
# rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
# Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
# sleep 1
# export DISPLAY=:99
hash -r

# 本机无 /usr/local/cuda；transformer_engine 需从 venv 的 nvidia pip 包加载 libnvrtc
# （否则 ldconfig | grep libnvrtc 找不到库会直接 CalledProcessError）
_VENV_SITE=/home/ubuntu/Dev/hermine/HIL-RL-Project/.venv/lib/python3.10/site-packages
export CUDA_HOME="${CUDA_HOME:-${_VENV_SITE}/nvidia}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export LD_LIBRARY_PATH="$(echo ${_VENV_SITE}/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH}"

# 绕过tokenizers path
export PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\n' | grep -v cosmos_policy | tr '\n' ':')
export LIBERO_CONFIG_PATH=/home/ubuntu/Dev/hermine/HIL-RL-Project/cosmos-policy/cosmos_policy/config/libero

# 指定保存权重路径
export IMAGINAIRE_OUTPUT_ROOT=/home/ubuntu/Dev/hermine/HIL-RL-Project/HIL-RL/experiments
# 指定huggingface保存的路径
export HF_HUB_CACHE=/home/ubuntu/Dev/hermine/HIL-RL-Project/cosmos-policy/cosmos_policy/models/Cosmos-Policy-LIBERO-Predict2-2B/

export PYTHONPATH=$PYTHONPATH:../../../cosmos-policy/
export PYTHONPATH=$PYTHONPATH:../../../cosmos-policy/cosmos_policy/
export PYTHONPATH=$PYTHONPATH:../../lerobot/src/
# export PYTHONPATH=$PYTHONPATH:../../../lerobot_new/src/
# export PYTHONPATH=$PYTHONPATH:../../../RL-Robot-Env/
export PYTHONPATH=$PYTHONPATH:../../rl_envs/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL


task_name=install_handle
mkdir -p experiments/${task_name}
cd experiments/${task_name}



# python3 ../../valid_space.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=false freeze_actor=false use_human_intervention=true ego_mode=true policy_type=silri
python3 ../../valid_space.py robot_type@_global_=tienyi task@_global_=${task_name} classifier_cfg.require_train=false freeze_actor=true use_human_intervention=true ego_mode=true policy_type=silri
