# v1_neg_OPSD 结果

模板负样本 pairwise 叠在 SFT CE 上：`L = L_CE + μ L_pref`（μ=0.3），负样本为 `The answer is {gold}` / `The final answer is {gold}` / `Sure, {gold}`。fresh LoRA。

InstrDialog++ 首次跑到 segment 27（`task1553_cnn_dailymail_summarization`）因 `token_mean_logprob` vocab 维 logsumexp CUDA OOM；已改为 fused CE nll 作 logp，且序列 ≥1024 token 时跳过 pairwise 只留 CE，整段重跑。

| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | Current-Seg Acc↑ | Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog | 0.346 | 0.362 | 0.089 | 0.083 | 0.430 | 0.425 | 0.451 | 0.422 | 0.000 | 0.000 | 0.332 | 对照起点 |
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog++ | 0.285 | 0.298 | 0.180 | 0.175 | 0.392 | 0.388 | 0.461 | 0.383 | 0.100 | 0.100 | 0.261 | 对照起点 |
| v1_neg_OPSD | 模板负样本 pairwise + CE（μ=0.3） | InstrDialog | 0.325 | 0.336 | 0.100 | 0.089 | 0.383 | 0.378 | 0.424 | 0.377 | 0.000 | 0.000 | 0.308 | 合理偏弱：Score 0.325 略低于 v0 的 0.346，F 0.100 接近；格式 pairwise 未超过纯 SFT，NLG 略掉 |
| v1_neg_OPSD | 模板负样本 pairwise + CE（μ=0.3） | InstrDialog++ | 0.337 | 0.358 | 0.113 | 0.102 | 0.425 | 0.420 | 0.484 | 0.419 | 0.000 | 0.000 | 0.310 | 合理且较强：Score 0.337 超过 v0 的 0.285 与 SFT+OPSD 的 0.300，F 0.113 优于 v0 的 0.180；CNN/DM（seg 27）OOM 修复后整段跑通。seg 30（CUAD）后出现一次 eval 全 0，下一段即恢复，最终指标正常 |

## run id

| 流 | run id | 文件 |
|---|---|---|
| InstrDialog | `20260826_002152_v1_neg_opsd_instrdialog` | `final_metrics_instrdialog.json` / `segment_metrics_instrdialog.csv` |
| InstrDialog++ | `20260826_203109_v1_neg_opsd_instrdialogpp`（OOM 修复后重跑） | `final_metrics_instrdialogpp.json` / `segment_metrics_instrdialogpp.csv` |
