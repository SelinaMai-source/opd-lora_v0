# v1_teacher_enhance_OPSD 结果

与 `v1_SFT_OPSD` 相同（加载 SFT adapter + JSD OPSD），teacher 过渡指令追加格式约束：`Preserve its key content, answer format, wording style, and approximate length.`

| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | Current-Seg Acc↑ | Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog | 0.346 | 0.362 | 0.089 | 0.083 | 0.430 | 0.425 | 0.451 | 0.422 | 0.000 | 0.000 | 0.332 | 对照起点 |
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog++ | 0.285 | 0.298 | 0.180 | 0.175 | 0.392 | 0.388 | 0.461 | 0.383 | 0.100 | 0.100 | 0.261 | 对照起点 |
| v1_SFT_OPSD | 加载 SFT adapter + OPSD（无格式约束） | InstrDialog++ | 0.300 | 0.313 | 0.084 | 0.082 | 0.409 | 0.402 | 0.457 | 0.412 | 0.000 | 0.200 | 0.272 | 无约束对照 |
| v1_teacher_enhance_OPSD | SFT init OPSD + teacher 格式约束 | InstrDialog | 0.153 | 0.189 | 0.194 | 0.178 | 0.253 | 0.245 | 0.321 | 0.259 | 0.000 | 0.000 | 0.171 | 负向：Score 0.153 远低于 v0 的 0.346 与无约束 SFT+OPSD；格式约束在 19 段流上伤害大，NLG 同步掉 |
| v1_teacher_enhance_OPSD | SFT init OPSD + teacher 格式约束 | InstrDialog++ | 0.244 | 0.249 | 0.139 | 0.139 | 0.379 | 0.373 | 0.428 | 0.387 | 0.000 | 0.100 | 0.220 | 负向：Score 低于无约束 SFT+OPSD（0.300）与 v0（0.285）；F 0.139 介于二者之间，格式约束未兑现「压修饰词」的收益 |

## run id

| 流 | run id | 文件 |
|---|---|---|
| InstrDialog | `20260825_194343_v1_teacher_enhance_opsd_instrdialog` | `final_metrics_instrdialog.json` / `segment_metrics_instrdialog.csv` |
| InstrDialog++ | `20260825_195501_v1_teacher_enhance_opsd_instrdialogpp` | `final_metrics_instrdialogpp.json` / `segment_metrics_instrdialogpp.csv` |
