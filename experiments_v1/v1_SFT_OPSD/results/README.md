# v1_SFT_OPSD 结果

加载 v0 SFT `final_adapter` 后继续 OPSD。InstrDialog 原 seed 123 末段崩盘；同 seed 重跑与 seed 456 均未崩。

| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | Current-Seg Acc↑ | Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog | 0.346 | 0.362 | 0.089 | 0.083 | 0.430 | 0.425 | 0.451 | 0.422 | 0.000 | 0.000 | 0.332 | 合理：SFT 基线，19 段流遗忘低（F 0.089），生成质量指标为全表最高档，作为 v1 对照起点 |
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog++ | 0.285 | 0.298 | 0.180 | 0.175 | 0.392 | 0.388 | 0.461 | 0.383 | 0.100 | 0.100 | 0.261 | 合理：38 段长流遗忘明显加重（F 0.180 vs 19 段 0.089），符合流越长越难抗遗忘的预期 |
| v1_SFT_OPSD | 加载 SFT 最终 adapter 初始化，继续 OPSD（原 run seed 123） | InstrDialog | 0.011 | 0.211 | 0.367 | 0.171 | 0.153 | 0.146 | 0.197 | 0.165 | 0.000 | 0.000 | 0.190 | 异常（单次）：末段 segment 18（task1600_smcalflow）训练后 strict EM 由段 17 末的 0.294 崩至 0.011（F 0.367）；2026-08-24 同 seed 重跑未崩、seed 456 亦未崩 → GPU 非确定性 + seed 敏感临界点，非该流系统性失稳 |
| v1_SFT_OPSD | 同 seed 123 重跑（每段保存 adapter） | InstrDialog | 0.263 | 0.283 | 0.182 | 0.172 | 0.361 | 0.353 | 0.396 | 0.366 | 0.000 | 0.000 | 0.256 | 未崩：段 18 后 Score 0.263（相对段 17 的 0.210 回升），无法复现原 0.011 崩盘 |
| v1_SFT_OPSD | seed 456 稳健性（同 SFT init adapter） | InstrDialog | 0.347 | 0.358 | 0.033 | 0.033 | 0.420 | 0.415 | 0.443 | 0.423 | 0.000 | 0.000 | 0.322 | 未崩且最优档：Score 0.347 与 v0 SFT 持平，F 0.033 为 InstrDialog 全表最低 |
| v1_SFT_OPSD | 加载 SFT 最终 adapter 初始化，继续 OPSD | InstrDialog++ | 0.300 | 0.313 | 0.084 | 0.082 | 0.409 | 0.402 | 0.457 | 0.412 | 0.000 | 0.200 | 0.272 | 合理且最优：Score 0.300 为长流当时最高，F 0.084 显著优于 v0 的 0.180——SFT 初始化 + OPSD 在长流上收益明确 |

## run id

| 流 | run id | 文件 |
|---|---|---|
| InstrDialog（原 run） | `20260820_181411_v1_sft_opsd_instrdialog` | `final_metrics_instrdialog.json` / `segment_metrics_instrdialog.csv` |
| InstrDialog（seed 123 重跑） | `v1_sft_opsd_instrdialog_seed123_rerun` | `final_metrics_instrdialog_seed123_rerun.json` / `segment_metrics_instrdialog_seed123_rerun.csv` |
| InstrDialog（seed 456） | `v1_sft_opsd_instrdialog_seed456` | `final_metrics_instrdialog_seed456.json` / `segment_metrics_instrdialog_seed456.csv` |
| InstrDialog++ | `20260820_182823_v1_sft_opsd_instrdialogpp` | `final_metrics_instrdialogpp.json` / `segment_metrics_instrdialogpp.csv` |
