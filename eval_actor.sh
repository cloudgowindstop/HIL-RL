#!/usr/bin/env bash
HIL_RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# source /media/HIL-RL-Project/HIL-RL/.venv/bin/activate
# export PATH="/media/HIL-RL-Project/HIL-RL/.venv/bin:$PATH"
source ${HIL_RL_ROOT}/../.venv/bin/activate
export PATH=${HIL_RL_ROOT}/../.venv/bin:$PATH
# source /media/jushen/linda-zhao/HIL-RL-Project/.venv/bin/activate
# export PATH=/media/jushen/linda-zhao/HIL-RL-Project/.venv/bin:$PATH

# source /home/eai/Dev/hermine/HIL-RL-Project/.venv/bin/activate
# # Xvfb 无物理键盘，pynput import 会阻塞；headless 评估需开启 NO_GUI
export NO_GUI=0
# # 清理 :99 残留锁文件，重启 Xvfb
# rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
# Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
# sleep 1
# export DISPLAY=:99

hash -r
# 绕过tokenizers path
export PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\n' | grep -v cosmos_policy | tr '\n' ':')
export LIBERO_CONFIG_PATH=${HIL_RL_ROOT}/../cosmos-policy/cosmos_policy/config/libero
# 指定huggingface保存的路径
export HF_HUB_CACHE=${HIL_RL_ROOT}/../cosmos-policy/cosmos_policy/models/Cosmos-Policy-LIBERO-Predict2-2B/
# export PYTHONPATH=$PYTHONPATH:../../lerobot-icml/src/
export PYTHONPATH=$PYTHONPATH:${HIL_RL_ROOT}/lerobot/src/
export PYTHONPATH=$PYTHONPATH:${HIL_RL_ROOT}/../cosmos-policy/
export PYTHONPATH=$PYTHONPATH:${HIL_RL_ROOT}/../cosmos-policy/cosmos_policy/
# export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/shiny/SoRLM/lerobot-icml/src/
# export PYTHONPATH=$PYTHONPATH:../../../RL-Robot-Env/
export PYTHONPATH=$PYTHONPATH:${HIL_RL_ROOT}/../rl_envs/
export PYTHONPATH=$PYTHONPATH:${HIL_RL_ROOT}
# export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs
# export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xtele

cd ${HIL_RL_ROOT}

unset http_proxy
unset https_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"
# export http_proxy=http://192.168.32.28:18000 && export https_proxy=http://192.168.32.28:18000
# export HF_ENDPOINT=https://huggingface.co
# export HF_HUB_ENABLE_HF_TRANSFER=0
# export http_proxy=http://127.0.0.1:7890 && export https_proxy=http://127.0.0.1:7890

# task_name=close_trashbin_franka_1028
# task_name=fold_rag_franka
# task_name=push_T_franka_1028
# task_name=hang_chinese_knot_franka_1028
# task_name=hang_chinese_knot_franka_conrft_regime
# task_name=insert_usb_franka_1028
# task_name=pick_toy
task_name=install_handle

mkdir -p experiments/${task_name}
cd experiments/${task_name}


# python3 ../../eval_actor.py robot_type@_global_=franka task@_global_=${task_name} \
#     policy_type=silri \
#     classifier_cfg.require_train=false \
#     load_path=\'/home/eai/Dev/linda/open_source_code/HIL-RL/experiments/close_trashbin_franka_1028/exp_local/2026.03.03/12973\'


# python3 ../../eval_actor.py robot_type@_global_=franka task@_global_=${task_name} \
#     policy_type=hgdagger \
#     classifier_cfg.require_train=false \
#     load_path=\'/home/eai/Dev/linda/open_source_code/test_ckpt/push_T/linda_policy_type=hgdagger_distribution,robot_type@_global_=franka2,task@_global_=push_T_franka_1028/checkpoints/022000/pretrained_model/22000\'

# python3 ../../eval_actor_copy.py robot_type@_global_=franka task@_global_=${task_name} \
#     policy_type=cosmos \
#     load_classifier=false \
#     classifier_cfg.require_train=false \
#     load_path=\'/media/HIL-RL-Project/HIL-RL/experiments/pick_toy/exp_local/2026.07.31/cosmos_1/\' \
#     use_human_intervention=false \
#     ego_mode=false
python3 ../../eval_actor_copy.py robot_type@_global_=tienyi task@_global_=${task_name} \
    policy_type=cosmos \
    load_classifier=false \
    classifier_cfg.require_train=false \
    load_path=\'/home/ubuntu/Dev/hermine/HIL-RL-Project/HIL-RL/experiments/install_handle/offline_ckpt/ \
    use_human_intervention=false \
    ego_mode=false

# /media/HIL-RL/.venv/bin/python

# python3 ../../eval_actor_cosmos.py robot_type@_global_=franka task@_global_=${task_name} \
#     policy_type=cosmos  \
#     classifier_cfg.require_train=false \
#     use_human_intervention=true \
#     ego_mode=true \


# python3 ../../eval_actor_cosmos_libero.py robot_type@_global_=franka task@_global_=${task_name} \
#     policy_type=cosmos  \
#     classifier_cfg.require_train=false \
#     use_human_intervention=false \
#     ego_mode=false