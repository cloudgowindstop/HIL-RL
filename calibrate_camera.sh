export PYTHONPATH=/home/ubuntu/Dev/hermine/HIL-RL-Project/HIL-RL/lerobot/src:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/Dev/hermine/HIL-RL-Project/HIL-RL/rl_envs
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/Dev/hermine/HIL-RL-Project/HIL-RL
# export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
unset http_proxy 
unset https_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# task_name=insert_tube
# task_name=tienyi_test
task_name=install_handle
# task_name=tienyi_test
mkdir -p experiments/${task_name}
cd experiments/${task_name}


python3 ../../calibrate_camera.py \
    robot_type@_global_=tienyi \
    task@_global_=${task_name} \
    "$@"
