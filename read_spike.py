import torch
import os
PATH = "/media/HIL-RL-Project/HIL-RL/experiments/close_trashbin_franka_1028/logs/demo_action_l1_spike/spike_00000790_to_00000791_dl0.2111"
obj = torch.load(os.path.join(PATH, "snapshot.pt"), map_location="cpu", weights_only=False)

prev, curr = obj["prev"], obj["curr"]
# 'optimization_step', 'demo_sample_action_l1_loss', 'forward_batch_all_tensors_cpu', 'wrist_image', 'primary_image', 'action', 'output_metrics'

# print(prev["optimization_step"], prev["demo_sample_action_l1_loss"])
# print(curr["optimization_step"], curr["demo_sample_action_l1_loss"])
# print(obj["loss_delta"])  # 应 > 0.2 才会生成该文件