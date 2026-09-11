# 清理 :99 残留锁文件，重启 Xvfb
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
sleep 1
export DISPLAY=:99
# 临时禁用conda path
export PATH=/media/HIL-RL/.venv/bin:/usr/bin:/bin
hash -r
# 绕过tokenizers path
export PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\n' | grep -v cosmos_policy | tr '\n' ':')

# 指定huggingface保存的路径
export HF_HUB_CACHE=/media/cosmos-policy/cosmos_policy/models/Cosmos-Policy-LIBERO-Predict2-2B/
export PYTHONPATH=$PYTHONPATH:../../../cosmos-policy/
export PYTHONPATH=$PYTHONPATH:../../../cosmos-policy/cosmos_policy/

export PYTHONPATH=$PYTHONPATH:../../lerobot/src/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL
export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
unset http_proxy
unset https_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"
# export http_proxy=http://127.0.0.1:8889 && export https_proxy=http://127.0.0.1:8889

task_name=close_trashbin_franka_1028
mkdir -p experiments/${task_name}
cd experiments/${task_name}


# python3 ../../actor.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=true use_human_intervention=true ego_mode=true policy_type=silri

# [debug]
# python3 ../../actor.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=true freeze_actor=true use_human_intervention=false ego_mode=false policy_type=silri

python3 ../../actor_dummy.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=true freeze_actor=false use_human_intervention=false ego_mode=false policy_type=cosmos