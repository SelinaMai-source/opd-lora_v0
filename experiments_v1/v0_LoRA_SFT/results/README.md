# v0_LoRA_SFT 结果

口径：Seen-Avg Acc = `eval.seen_avg_score`（strict EM），Forgetting = `eval.forgetting`，数值保留 3 位小数。全量 pred/gold CSV 见 `../predictions/`（不入库）。

| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | Current-Seg Acc↑ | Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog | 0.346 | 0.362 | 0.089 | 0.083 | 0.430 | 0.425 | 0.451 | 0.422 | 0.000 | 0.000 | 0.332 | 合理：SFT 基线，19 段流遗忘低（F 0.089），生成质量指标为全表最高档，作为 v1 对照起点 |
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog++ | 0.285 | 0.298 | 0.180 | 0.175 | 0.392 | 0.388 | 0.461 | 0.383 | 0.100 | 0.100 | 0.261 | 合理：38 段长流遗忘明显加重（F 0.180 vs 19 段 0.089），符合流越长越难抗遗忘的预期 |

## run id

| 流 | run id | 文件 |
|---|---|---|
| InstrDialog | `v0_sft_instrdialog` | `final_metrics_instrdialog.json` / `segment_metrics_instrdialog.csv` |
| InstrDialog++ | `v0_sft_instrdialogpp` | `final_metrics_instrdialogpp.json` / `segment_metrics_instrdialogpp.csv` |
