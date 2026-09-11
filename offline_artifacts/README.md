# offline_artifacts

Light training logs and 2000-step validation from `action_representation_20260903`. No `*.pt` checkpoints.

| Path | Contents |
| --- | --- |
| `compare_matrix_2000/` | Main 2000-step compare tables |
| `compare_matrix/` | 1000-step appendix |
| `value_spearman_2000/` | Value Spearman eval (n=80) |
| `metrics/` | Per-run `metrics.jsonl` |
| `logs/` | WandB `output.log` / summary JSON for 2000-step runs |
| `visualize/` | World RGB contact sheets (png) + mp4 for the four finished 2000-step runs |
| `eval_logs/` | Small eval logs from `train_outputs/` |

Writeup and plots: [`../mingbo_result/README.md`](../mingbo_result/README.md).
