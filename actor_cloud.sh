# 临时禁用conda path
source /media/HIL-RL-Project/HIL-RL/.venv/bin/activate
export PATH=/media/HIL-RL-Project/HIL-RL/.venv/bin:$PATH

# 清理 :99 残留锁文件，重启 Xvfb
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
sleep 1
export DISPLAY=:99

hash -r
# 绕过tokenizers path
export PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\n' | grep -v cosmos_policy | tr '\n' ':')

# 指定保存权重路径
export IMAGINAIRE_OUTPUT_ROOT=/media/HIL-RL-Project/HIL-RL/experiments
# 指定huggingface保存的路径
export HF_HUB_CACHE=/media/cosmos-policy/cosmos_policy/models/Cosmos-Policy-LIBERO-Predict2-2B/

export PYTHONPATH=$PYTHONPATH:../../../cosmos-policy/
export PYTHONPATH=$PYTHONPATH:../../../cosmos-policy/cosmos_policy/
export PYTHONPATH=$PYTHONPATH:../../lerobot/src/
# export PYTHONPATH=$PYTHONPATH:../../../lerobot_new/src/
# export PYTHONPATH=$PYTHONPATH:../../../RL-Robot-Env/
export PYTHONPATH=$PYTHONPATH:../../rl_envs/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL
export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
# export http_proxy=http://127.0.0.1:7890 && export https_proxy=http://127.0.0.1:7890
unset http_proxy
unset https_proxy
# export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export no_proxy=localhost,127.0.0.1,0.0.0.0,::1,192.168.0.0/16
export NO_PROXY=$no_proxy


# task_name=close_trashbin_franka_1028
task_name=pick_toy
# task_name=fold_rag_franka
# task_name=push_T
mkdir -p experiments/${task_name}
cd experiments/${task_name}


# python3 ../../actor.py robot_type@_global_=franka task@_global_=${task_name} \
#     classifier_cfg.require_train=true use_human_intervention=true ego_mode=true policy_type=rlif_lag_single_bc \
#     load_path=\'/home/eai/Dev/linda/open_source_code/HIL-RL/experiments/fold_rag_franka/exp_local/2026.02.05/classifier_cfg.require_train=true,ego_mode=true,policy_type=rlif_lag_single_bc,robot_type@_global_=franka,task@_global_=fold_rag_franka,use_human_intervention=true/checkpoints/025193/pretrained_model/25193\'

# python3 ../../actor.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=true use_human_intervention=true ego_mode=false policy_type=calql_sac_parl

# python3 ../../actor.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=true use_human_intervention=true ego_mode=false policy_type=calql_sac_parl
python3 ../../actor_cloud.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=false use_human_intervention=false ego_mode=false policy_type=cosmos
# python3 ../../actor.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=true use_human_intervention=true ego_mode=true policy_type=sac

# [debug]
# python3 ../../actor.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=true freeze_actor=true use_human_intervention=false ego_mode=false policy_type=silri