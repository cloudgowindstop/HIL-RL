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
# export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/
# export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xrocs/

# source /home/ubuntu/XARM/install/setup.bash

# python3 -m pip install xrocs-3.1.6-cp310-cp310-linux_x86_64.whl
# python3 -m pip uninstall xtele
# curl -fsSL http://10.20.48.5:18080/install.sh | bash -s -- -V 2.4.12 --python 3.10

# export http_proxy=http://127.0.0.1:7890 && export https_proxy=http://127.0.0.1:7890
unset http_proxy
unset https_proxy
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"
# export http_proxy=http://192.168.32.28:18000
# export https_proxy=http://192.168.32.28:18000
export no_proxy=localhost,127.0.0.1,0.0.0.0,::1,192.168.0.0/16
export NO_PROXY=$no_proxy

# task_name=close_trashbin_franka_1028
# task_name=fold_rag_franka
# task_name=push_T
# task_name=pick_toy
task_name=install_handle
mkdir -p experiments/${task_name}
cd experiments/${task_name}


export ENCODE_CUDA_DEVICES=0



# python3 ../../collect_data.py robot_type@_global_=franka task@_global_=${task_name} use_human_intervention=true ego_mode=true load_classifier=false
# python3 ../../collect_data.py robot_type@_global_=ur task@_global_=${task_name} use_human_intervention=true ego_mode=false load_classifier=false
# python3 ../../collect_data_cosmos.py robot_type@_global_=franka task@_global_=${task_name} use_human_intervention=false ego_mode=false load_classifier=false policy_type=cosmos
# python3 ../../collect_data_cosmos.py robot_type@_global_=franka task@_global_=${task_name} use_human_intervention=true ego_mode=true load_classifier=false policy_type=cosmos
# python3 ../../collect_data_cosmos_save_origin.py robot_type@_global_=franka task@_global_=${task_name} use_human_intervention=true ego_mode=true load_classifier=false policy_type=cosmos
python3 ../../collect_data_cosmos.py robot_type@_global_=tienyi task@_global_=${task_name} use_human_intervention=true ego_mode=true load_classifier=false policy_type=cosmos
