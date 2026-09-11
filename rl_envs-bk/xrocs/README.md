# XROCS: A Unified Interface for Robot Control 

## Introduction 
XROCS is a unified, well-structured interface for robot control, covering the robot arm, camera, and gripper. We currently open-source interfaces for **UR5**, **Orbbec cameras**, and the **Robotiq gripper**, released alongside the paper [SiLRI](https://arxiv.org/pdf/2512.24288). However, xrocs can be easy to override and extend to other hardware setups. 



## Get Started 

To install `xrocs`, run: 
```bash 
git clone https://github.com/nuomizai/xrocs/ 
cd xrocs 
pip install -e . 
``` 

Then run the example: 

```bash 
python3 ur_station_example.py 
``` 

For more usage examples, see `base_env.py` in [rl_envs](https://github.com/nuomizai/rl_envs) and the full project [HIL-RL](https://github.com/nuomizai/HIL-RL). 

## FAQs 
If you run into issues, please open a GitHub issue or contact `linda.chao.007@gmail.com`. Feedback and contributions are welcome.