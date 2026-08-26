# v1_OPSD 结果

fresh LoRA 直接 OPSD（无 SFT 初始化）。对照 v0 见本表前两行。

| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | Current-Seg Acc↑ | Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog | 0.346 | 0.362 | 0.089 | 0.083 | 0.430 | 0.425 | 0.451 | 0.422 | 0.000 | 0.000 | 0.332 | 合理：SFT 基线，19 段流遗忘低（F 0.089），生成质量指标为全表最高档，作为 v1 对照起点 |
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog++ | 0.285 | 0.298 | 0.180 | 0.175 | 0.392 | 0.388 | 0.461 | 0.383 | 0.100 | 0.100 | 0.261 | 合理：38 段长流遗忘明显加重（F 0.180 vs 19 段 0.089），符合流越长越难抗遗忘的预期 |
| v1_OPSD | fresh LoRA 直接 OPSD，无 SFT 初始化 | InstrDialog | 0.289 | 0.299 | 0.128 | 0.133 | 0.361 | 0.357 | 0.385 | 0.357 | 0.000 | 0.000 | 0.270 | 合理但偏弱：训练稳定无崩盘、F 与 v0 相当，但 Score 低于 SFT 约 5.8pp——仅 on-policy 蒸馏不足以替代 SFT 起点 |
| v1_OPSD | fresh LoRA 直接 OPSD，无 SFT 初始化 | InstrDialog++ | 0.223 | 0.255 | 0.171 | 0.154 | 0.359 | 0.352 | 0.423 | 0.370 | 0.100 | 0.300 | 0.224 | 合理但偏弱：同样稳定无崩盘，Score 低于 v0 约 6.2pp、F 与 v0 接近，结论与 19 段流一致 |

## run id

| 流 | run id | 文件 |
|---|---|---|
| InstrDialog | `20260820_172934_v1_opsd_instrdialog` | `final_metrics_instrdialog.json` / `segment_metrics_instrdialog.csv` |
| InstrDialog++ | `20260820_174143_v1_opsd_instrdialogpp` | `final_metrics_instrdialogpp.json` / `segment_metrics_instrdialogpp.csv` |
