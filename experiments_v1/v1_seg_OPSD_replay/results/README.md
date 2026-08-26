# v1_seg_OPSD_replay 结果

在胜出 K=25 的段内 SFT/OPSD 交替上，当前段 batch ≈ 80% 当前段 + 20% 已见段均匀抽样。

| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | Current-Seg Acc↑ | Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog | 0.346 | 0.362 | 0.089 | 0.083 | 0.430 | 0.425 | 0.451 | 0.422 | 0.000 | 0.000 | 0.332 | 对照起点 |
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog++ | 0.285 | 0.298 | 0.180 | 0.175 | 0.392 | 0.388 | 0.461 | 0.383 | 0.100 | 0.100 | 0.261 | 对照起点 |
| v1_seg_OPSD | 段内交替 K=25，无 replay | InstrDialog | 0.289 | 0.304 | 0.156 | 0.156 | 0.376 | 0.372 | 0.415 | 0.371 | 0.000 | 0.000 | 0.280 | 无 replay 对照 |
| v1_seg_OPSD | 段内交替 K=25，无 replay | InstrDialog++ | 0.289 | 0.305 | 0.151 | 0.146 | 0.404 | 0.395 | 0.466 | 0.394 | 0.000 | 0.000 | 0.267 | 无 replay 对照 |
| v1_seg_OPSD_replay | 段内交替 K=25 + 20% replay | InstrDialog | 0.378 | 0.389 | 0.083 | 0.072 | 0.443 | 0.440 | 0.463 | 0.435 | 0.100 | 0.100 | 0.355 | 本轮最强短流：Score 0.378 超过 v0 的 0.346，F 0.083 与 v0 持平；20% replay 明显补上交替本身的遗忘 |
| v1_seg_OPSD_replay | 段内交替 K=25 + 20% replay | InstrDialog++ | 0.379 | 0.381 | 0.108 | 0.111 | 0.466 | 0.461 | 0.517 | 0.460 | 0.200 | 0.200 | 0.330 | 本轮最强长流：Score 0.379 超过 v0 的 0.285 与 SFT+OPSD 的 0.300；F 0.108 优于 v0 的 0.180；NLG（F1/ROUGE/BLEU）同为长流最高档 |

## run id

| 流 | run id | 文件 |
|---|---|---|
| InstrDialog | `20260825_223107_v1_seg_opsd_replay_instrdialog` | `final_metrics_instrdialog.json` / `segment_metrics_instrdialog.csv` |
| InstrDialog++ | `20260825_230728_v1_seg_opsd_replay_instrdialogpp` | `final_metrics_instrdialogpp.json` / `segment_metrics_instrdialogpp.csv` |
