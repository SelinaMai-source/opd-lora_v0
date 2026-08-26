# v1_gold_OPSD 结果

每步 `L = L_CE + λ L_KL`（λ=0.3）：gold 上 SFT CE + 冻结基座 teacher 在同一 gold token 上 KL(p_T ∥ p_S)。fresh LoRA，开启 teacher 格式约束。

| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | Current-Seg Acc↑ | Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog | 0.346 | 0.362 | 0.089 | 0.083 | 0.430 | 0.425 | 0.451 | 0.422 | 0.000 | 0.000 | 0.332 | 对照起点 |
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog++ | 0.285 | 0.298 | 0.180 | 0.175 | 0.392 | 0.388 | 0.461 | 0.383 | 0.100 | 0.100 | 0.261 | 对照起点 |
| v1_gold_OPSD | 同一步 CE + gold KL（λ=0.3） | InstrDialog | 0.341 | 0.357 | 0.106 | 0.089 | 0.403 | 0.400 | 0.430 | 0.399 | 0.000 | 0.000 | 0.327 | 合理：与 v0 SFT 几乎持平（0.341 vs 0.346），符合「主项仍是 CE」的设计；F 略差（0.106 vs 0.089） |
| v1_gold_OPSD | 同一步 CE + gold KL（λ=0.3） | InstrDialog++ | 0.303 | 0.316 | 0.157 | 0.147 | 0.402 | 0.397 | 0.464 | 0.394 | 0.100 | 0.100 | 0.276 | 合理：Score 0.303 略高于 v0 的 0.285、与 SFT+OPSD 的 0.300 接近；F 0.157 优于 v0 的 0.180，但弱于 SFT+OPSD 的 0.084 |

## run id

| 流 | run id | 文件 |
|---|---|---|
| InstrDialog | `20260825_235819_v1_gold_opsd_instrdialog` | `final_metrics_instrdialog.json` / `segment_metrics_instrdialog.csv` |
| InstrDialog++ | `20260826_000404_v1_gold_opsd_instrdialogpp` | `final_metrics_instrdialogpp.json` / `segment_metrics_instrdialogpp.csv` |
