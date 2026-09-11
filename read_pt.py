"""对比 spike 落盘的 x0（真值 latent）与 model_pred_x0（预测 latent）。"""
import torch

PATH1 = "/media/HIL-RL-Project/HIL-RL/experiments/0511_cosmos_demo_action_l1_csv/spike_step_000000994_delta_mean_p0d218836/max_x0.pt"
PATH2 = "/media/HIL-RL-Project/HIL-RL/experiments/0511_cosmos_demo_action_l1_csv/spike_step_000000994_delta_mean_p0d218836/max_model_pred_x0.pt"
PATH3 = "/media/HIL-RL-Project/HIL-RL/experiments/0511_cosmos_demo_action_l1_csv/spike_step_000000994_delta_mean_p0d218836/prev_max_x0.pt"
PATH4 = "/media/HIL-RL-Project/HIL-RL/experiments/0511_cosmos_demo_action_l1_csv/spike_step_000000994_delta_mean_p0d218836/prev_max_model_pred_x0.pt"
PATH5 = "/media/HIL-RL-Project/HIL-RL/experiments/0511_cosmos_demo_action_l1_csv/spike_step_000000994_delta_mean_p0d218836/min_x0.pt"
PATH6 = "/media/HIL-RL-Project/HIL-RL/experiments/0511_cosmos_demo_action_l1_csv/spike_step_000000994_delta_mean_p0d218836/min_model_pred_x0.pt"
PATH7 = "/media/HIL-RL-Project/HIL-RL/experiments/0511_cosmos_demo_action_l1_csv/spike_step_000000994_delta_mean_p0d218836/prev_min_x0.pt"
PATH8 = "/media/HIL-RL-Project/HIL-RL/experiments/0511_cosmos_demo_action_l1_csv/spike_step_000000994_delta_mean_p0d218836/prev_min_model_pred_x0.pt"

def _load(path: str) -> torch.Tensor:
    kw = {"map_location": "cpu"}
    try:
        t = torch.load(path, weights_only=True, **kw)
    except TypeError:
        t = torch.load(path, **kw)
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"期望 Tensor，得到 {type(t)}")
    return t.float()


def main() -> None:
    x0 = _load(PATH1)
    pred = _load(PATH2)
    x02 = _load(PATH3)
    pred2 = _load(PATH4)
    x03 = _load(PATH5)
    pred3 = _load(PATH6)
    x04 = _load(PATH7)
    pred4 = _load(PATH8)
    print("当前步max样本的真实值：", x0)
    print("当前步max样本的预测值：", pred)
    print("上一时刻max样本的真实值：", x02)
    print("上一时刻max样本的预测值：", pred2)

    # print("=== 基本信息 ===")
    # print("当前步max样本x0 (真值):     shape", tuple(x0.shape), "dtype", x0.dtype, "mean", x0.mean().item())
    # print("当前步max样本pred (预测):  shape", tuple(pred.shape), "dtype", pred.dtype, "mean", pred.mean().item())
    # print("上一时刻max样本x0 (真值):     shape", tuple(x02.shape), "dtype", x02.dtype, "mean", x02.mean().item())
    # print("上一时刻max样本pred (预测):  shape", tuple(pred2.shape), "dtype", pred2.dtype, "mean", pred2.mean().item())
    # print("当前步min样本x0 (真值):     shape", tuple(x03.shape), "dtype", x03.dtype, "mean", x03.mean().item())
    # print("当前步min样本pred (预测):  shape", tuple(pred3.shape), "dtype", pred3.dtype, "mean", pred3.mean().item())
    # print("上一时刻min样本x0 (真值):     shape", tuple(x04.shape), "dtype", x04.dtype, "mean", x04.mean().item())
    # print("上一时刻min样本pred (预测):  shape", tuple(pred4.shape), "dtype", pred4.dtype, "mean", pred4.mean().item())

    # if x0.shape != pred.shape:
    #     print("\n形状不一致，无法逐元素对比。可对齐后再比（如插值或裁切）。")
    #     return

    # diff = pred - x0
    # abs_diff = diff.abs()

    # diff2 = pred2 - x02
    # abs_diff2 = diff2.abs()

    # diff3 = pred3 - x03
    # abs_diff3 = diff3.abs()

    # diff4 = pred4 - x04
    # abs_diff4 = diff4.abs()

    # print("当前步max样本的x0与pred的差异：", abs_diff.mean().item())
    # print("上一时刻max样本的x0与pred的差异：", abs_diff2.mean().item())
    # print("当前步min样本的x0与pred的差异：", abs_diff3.mean().item())
    # print("上一时刻min样本的x0与pred的差异：", abs_diff4.mean().item())

if __name__ == "__main__":
    main()
