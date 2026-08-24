# v1_SFT_OPSD：SFT 初始化 + OPSD

## 设置

- **方法**：与 v1_OPSD 相同的 OPSD 训练目标（on-policy rollout + 双 forward + 逐 token JSD β=0.5），唯一差异是**首个 segment 前通过 `model.init_adapter_path` 加载 v0 SFT 重跑保存的最终 adapter**，在 SFT 基础上继续做 OPSD。
- **SFT adapter 来源**：`results/runs/v0_sft_instrdialog/final_adapter`、`results/runs/v0_sft_instrdialogpp/final_adapter`（不入库，需按下方命令先重跑 SFT）
- **其余口径**：与 v0/v1_OPSD 完全一致（r=16/α=32/q,v_proj/lr 2e-4/bs 2/每段 1 epoch/seed 123，Llama-3.1-8B-Instruct bfloat16）
- **配置**：`configs/` 内为本变体两条流的 yaml 副本

## 结果（全指标，提取自各 run 的 final_metrics.json，数值保留 3 位小数）

| Method | Data | Output Length | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_LoRA_SFT | InstrDialog | 10.3 | 0.346 | 0.362 | 0.089 | 0.083 | 0.430 | 0.425 | 0.451 | 0.422 | 0.000 | 0.000 | 0.332 | 合理：SFT 基线，19 段流遗忘低（F 0.089），生成质量指标为全表最高档，作为 v1 对照起点 |
| v0_LoRA_SFT | InstrDialog++ | 5.0 | 0.285 | 0.298 | 0.180 | 0.175 | 0.392 | 0.388 | 0.461 | 0.383 | 0.100 | 0.100 | 0.261 | 合理：38 段长流遗忘明显加重（F 0.180 vs 19 段 0.089），符合流越长越难抗遗忘的预期 |
| v1_OPSD | InstrDialog | 9.7 | 0.289 | 0.299 | 0.128 | 0.133 | 0.361 | 0.357 | 0.385 | 0.357 | 0.000 | 0.000 | 0.270 | 合理但偏弱：训练稳定无崩盘、F 与 v0 相当，但 Score 低于 SFT 约 5.8pp——仅 on-policy 蒸馏不足以替代 SFT 起点 |
| v1_OPSD | InstrDialog++ | 9.3 | 0.223 | 0.255 | 0.171 | 0.154 | 0.359 | 0.352 | 0.423 | 0.370 | 0.100 | 0.300 | 0.224 | 合理但偏弱：同样稳定无崩盘，Score 低于 v0 约 6.2pp、F 与 v0 接近，结论与 19 段流一致 |
| v1_SFT_OPSD | InstrDialog | 14.7 | 0.011 | 0.211 | 0.367 | 0.171 | 0.153 | 0.146 | 0.197 | 0.165 | 0.000 | 0.000 | 0.190 | 异常：末段（segment 18, task1600_smcalflow）训练后 strict EM 由段 17 末的 0.294 崩至 0.011（F 0.367），Token F1/ROUGE-L/BLEU 同步腰斩且输出变长失控（均长 14.7、最长打满 64 token）；训练侧 loss/grad_norm 正常、同段 v0/v1_OPSD 均不崩，已排除 bug，属真实剧烈遗忘 |
| v1_SFT_OPSD | InstrDialog++ | 7.7 | 0.300 | 0.313 | 0.084 | 0.082 | 0.409 | 0.402 | 0.457 | 0.412 | 0.000 | 0.200 | 0.272 | 合理且最优：Score 0.300 为全表最高，F 0.084 显著优于 v0 的 0.180——SFT 初始化 + OPSD 在长流上收益明确 |

注：Seen-Avg Acc = `eval.seen_avg_score`（strict EM），Forgetting = `eval.forgetting`，Current-Seg Acc = `eval.current_score`（末段），Task-aware Score Mean = `eval.task_aware_score_mean`（训练轨迹均值）；Output Length 取自最终评估点 eval_debug dump（前 50 条子集）`raw_generated_output` 的平均 token 数（Llama-3.1 tokenizer，不含 special tokens），final_metrics 无此字段。完整口径见 `docs/v1_experiment_report.md`（/root/autodl-tmp/docs/）。

## 末段崩盘诊断（InstrDialog 流）

- segment 17 结束时 seen_avg strict EM 仍为 **0.2944**（F 0.088），segment 18（`task1600_smcalflow_sentence_generation`）训练后崩至 **0.0105**（F 0.367），token F1/ROUGE-L/BLEU 同步腰斩。
- 训练侧指标正常（末段 JSD loss 0.117、grad_norm 1.02、rollout 均长 13.26），无 NaN/爆炸，**排除实现 bug**。
- 同段对照：v0 SFT 与 v1_OPSD 在 segment 18 均正常（0.3465 / 0.2886），**排除数据段异常**。
- 崩塌为已见 19 段**全局退化**（非仅当前段得 0），属真实灾难性遗忘。逐段轨迹见 `results/segment_metrics_instrdialog.csv`。

## run id 与产物

| 流 | run id | 配置 |
|---|---|---|
| InstrDialog（19 段） | `20260820_181411_v1_sft_opsd_instrdialog` | v1_sft_opsd_instrdialog.yaml |
| InstrDialog++（38 段） | `20260820_182823_v1_sft_opsd_instrdialogpp` | v1_sft_opsd_instrdialogpp.yaml |

- `results/`：两条流的 `final_metrics_*.json` 与 `segment_metrics_*.csv`
- 全量运行目录：`results/runs/<run_id>/`

## 复现

```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 先重跑 SFT 产出 adapter（output.save_final_adapter: true 已开启）
python -m core.train --config configs/v0_sequential_instrdialog.yaml     # run_name: v0_sft_instrdialog
python -m core.train --config configs/v0_sequential_instrdialogpp.yaml   # run_name: v0_sft_instrdialogpp
# 再做 SFT 初始化 OPSD
python -m core.train --config configs/v1_sft_opsd_instrdialog.yaml
python -m core.train --config configs/v1_sft_opsd_instrdialogpp.yaml
```
