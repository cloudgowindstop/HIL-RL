# rl_envs: A Real-world RL Wrapper for Robot Manipulation

## Introduction

`rl_envs` is a unified wrapper for real-world reinforcement learning. It provides sensor data collection, robot (arm and gripper) control, and a human intervention interface behind a Gym-style API—so you can train agents in the physical world as if you were running a simulator.

Currently, we open-source interfaces for **UR5**, **Orbbec cameras**, and the **Robotiq gripper**, released alongside [SiLRI](https://arxiv.org/pdf/2512.24288). The framework is modular and can be extended to other platforms (e.g., Franka or dual-arm systems).

## Get Started

### Step 1: Clone the repo

```bash
git clone https://github.com/nuomizai/rl_envs/
cd rl_envs
```

### Step 2: Install xrocs

**(A) Configure hardware.** Update the settings in `xrocs/configuration.toml` (e.g., IP address, camera names) to match your setup.

**(B) Install.**

```bash
cd xrocs && pip install -e . && cd ..
```

**(C) Validate.**
Start the camera node and the UR robot, then run:
```bash
python3 ur_station_example.py
```

### Step 3: Install xtele

**(A) Configure teleop.** Update `xtele/xtele/examples/ur_single.toml` if needed (e.g., `home_pose`; usually no change is required).

**(B) Install dependencies and scripts.**

```bash
cd xtele
pip install -e .
cd xtele/scripts
bash install_all.sh
cd ../../../
```

**(C) Validate.**

```bash
# (Optional) check teleoperation status
xhumanoid-xtele --mode getstates

# (Required) calibrate for real-time sync between teleop agent and robot
xhumanoid-xtele --mode cali
```

For more examples, see `base_env.py`.

## Contributors

This project is primarily maintained by Yinuo Zhao ([@nuomizai](https://github.com/nuomizai)), Junjie Ji ([@jasonm5a526](https://github.com/jasonm5a526)), and Pei Ren ([@im-renpei](https://github.com/im-renpei)).


## FAQs

If you run into issues, please open a GitHub issue or contact `linda.chao.007@gmail.com`. Feedback and contributions are welcome.

