# v1_seg_OPSD 结果

段内按 K 个 optimizer step 交替：SFT K 步 → 冻结当前 LoRA 为 teacher → OPSD K 步。InstrDialog 网格 K∈{25,50,75}（约 1/2/3 epoch/phase），胜出 K=25 打 InstrDialog++。

| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | Current-Seg Acc↑ | Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog | 0.346 | 0.362 | 0.089 | 0.083 | 0.430 | 0.425 | 0.451 | 0.422 | 0.000 | 0.000 | 0.332 | 对照起点 |
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog++ | 0.285 | 0.298 | 0.180 | 0.175 | 0.392 | 0.388 | 0.461 | 0.383 | 0.100 | 0.100 | 0.261 | 对照起点 |
| v1_seg_OPSD | 段内 SFT/OPSD 交替，K=25（约 1 epoch/phase） | InstrDialog | 0.289 | 0.304 | 0.156 | 0.156 | 0.376 | 0.372 | 0.415 | 0.371 | 0.000 | 0.000 | 0.280 | 网格最优：三条 K 中 Score 最高、F 最低；仍低于 v0 SFT（0.346 / 0.089） |
| v1_seg_OPSD | 段内 SFT/OPSD 交替，K=50（约 2 epoch/phase） | InstrDialog | 0.268 | 0.283 | 0.183 | 0.178 | 0.349 | 0.343 | 0.388 | 0.345 | 0.000 | 0.000 | 0.261 | 更长 phase 未带来收益，Score/F 均差于 K=25 |
| v1_seg_OPSD | 段内 SFT/OPSD 交替，K=75（约 3 epoch/phase） | InstrDialog | 0.205 | 0.247 | 0.232 | 0.199 | 0.285 | 0.278 | 0.343 | 0.279 | 0.100 | 0.100 | 0.227 | 最差：K 越大过拟合风险越高，Score 掉到 0.205 |
| v1_seg_OPSD | 胜出 K=25 打长流 | InstrDialog++ | 0.289 | 0.305 | 0.151 | 0.146 | 0.404 | 0.395 | 0.466 | 0.394 | 0.000 | 0.000 | 0.267 | 合理：Score 与 v0 持平（0.289 vs 0.285），F 0.151 优于 v0 的 0.180，但不如 SFT+OPSD 的 0.084 |

网格选取：优先 Seen-Avg Acc，同分看 Forgetting。胜出 **K=25**。

## run id

| 流 | run id | 文件 |
|---|---|---|
| InstrDialog K=25 | `20260825_202756_v1_seg_opsd_instrdialog_k25` | `final_metrics_instrdialog_k25.json` / `segment_metrics_instrdialog_k25.csv` |
| InstrDialog K=50 | `20260825_204314_v1_seg_opsd_instrdialog_k50` | `final_metrics_instrdialog_k50.json` / `segment_metrics_instrdialog_k50.csv` |
| InstrDialog K=75 | `20260825_211000_v1_seg_opsd_instrdialog_k75` | `final_metrics_instrdialog_k75.json` / `segment_metrics_instrdialog_k75.csv` |
| InstrDialog++ K=25 | `20260825_215010_v1_seg_opsd_instrdialogpp` | `final_metrics_instrdialogpp.json` / `segment_metrics_instrdialogpp.csv` |
