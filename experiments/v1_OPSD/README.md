# v1_OPSD：fresh LoRA 直接 OPSD（无 SFT 初始化）

## 设置

- **方法**：OPSD（on-policy self-distillation）。每 batch：当前 student 采样 rollout（temperature=1.0）→ 同一 rollout 接在 student prompt 与 teacher prompt（含 privileged reference `y*` + transition instruction）后做双 forward → rollout 逐 token generalized JSD（β=0.5）→ 只更新 student adapter。Teacher 为冻结基座（PEFT disable_adapter，不占双份显存）。
- **初始化**：fresh LoRA（r=16, alpha=32, dropout=0.05, q/v_proj, lr 2e-4, bs 2, 每段 1 epoch, seed 123）
- **Base Model**：Meta-Llama-3.1-8B-Instruct（bfloat16）
- **实现代码**：`baselines/basic_baselines/opsd/method.py`、`core/formatting.py`（format_for_teacher）、`core/models/base_model.py`（sample_rollouts / forward_logits）
- **配置**：`configs/` 内为本变体两条流的 yaml 副本（与主 `configs/` 一致）

## 结果（全指标，提取自各 run 的 final_metrics.json，数值保留 3 位小数）

| 版本 | 内容 | Data | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog | 0.346 | 0.362 | 0.089 | 0.083 | 0.430 | 0.425 | 0.451 | 0.422 | 0.000 | 0.000 | 0.332 | 合理：SFT 基线，19 段流遗忘低（F 0.089），生成质量指标为全表最高档，作为 v1 对照起点 |
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog++ | 0.285 | 0.298 | 0.180 | 0.175 | 0.392 | 0.388 | 0.461 | 0.383 | 0.100 | 0.100 | 0.261 | 合理：38 段长流遗忘明显加重（F 0.180 vs 19 段 0.089），符合流越长越难抗遗忘的预期 |
| v1_OPSD | fresh LoRA 直接 OPSD，无 SFT 初始化 | InstrDialog | 0.289 | 0.299 | 0.128 | 0.133 | 0.361 | 0.357 | 0.385 | 0.357 | 0.000 | 0.000 | 0.270 | 合理但偏弱：训练稳定无崩盘、F 与 v0 相当，但 Score 低于 SFT 约 5.8pp——仅 on-policy 蒸馏不足以替代 SFT 起点 |
| v1_OPSD | fresh LoRA 直接 OPSD，无 SFT 初始化 | InstrDialog++ | 0.223 | 0.255 | 0.171 | 0.154 | 0.359 | 0.352 | 0.423 | 0.370 | 0.100 | 0.300 | 0.224 | 合理但偏弱：同样稳定无崩盘，Score 低于 v0 约 6.2pp、F 与 v0 接近，结论与 19 段流一致 |
| v1_SFT_OPSD | 加载 SFT 最终 adapter 初始化，继续 OPSD | InstrDialog | 0.011 | 0.211 | 0.367 | 0.171 | 0.153 | 0.146 | 0.197 | 0.165 | 0.000 | 0.000 | 0.190 | 异常：末段（segment 18, task1600_smcalflow）训练后 strict EM 由段 17 末的 0.294 崩至 0.011（F 0.367），Token F1/ROUGE-L/BLEU 同步腰斩且输出变长失控（均长 14.7、最长打满 64 token）；训练侧 loss/grad_norm 正常、同段 v0/v1_OPSD 均不崩，已排除 bug，属真实剧烈遗忘 |
| v1_SFT_OPSD | 加载 SFT 最终 adapter 初始化，继续 OPSD | InstrDialog++ | 0.300 | 0.313 | 0.084 | 0.082 | 0.409 | 0.402 | 0.457 | 0.412 | 0.000 | 0.200 | 0.272 | 合理且最优：Score 0.300 为全表最高，F 0.084 显著优于 v0 的 0.180——SFT 初始化 + OPSD 在长流上收益明确 |

注：Seen-Avg Acc = `eval.seen_avg_score`（strict EM），Forgetting = `eval.forgetting`，Current-Seg Acc = `eval.current_score`（末段），Task-aware Score Mean = `eval.task_aware_score_mean`（训练轨迹均值）；Output Length 指标已从表中移除，崩盘行输出失控现象见结果解释。完整口径与崩盘诊断见 `docs/v1_experiment_report.md`（/root/autodl-tmp/docs/）。

## run id 与产物

| 流 | run id | 配置 |
|---|---|---|
| InstrDialog（19 段） | `20260820_172934_v1_opsd_instrdialog` | v1_opsd_instrdialog.yaml |
| InstrDialog++（38 段） | `20260820_174143_v1_opsd_instrdialogpp` | v1_opsd_instrdialogpp.yaml |

- `results/`：两条流的 `final_metrics_*.json` 与 `segment_metrics_*.csv`（逐段指标）
- 全量运行目录（含 eval_debug）：`results/runs/<run_id>/`

## 复现

```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m core.train --config configs/v1_opsd_instrdialog.yaml
python -m core.train --config configs/v1_opsd_instrdialogpp.yaml
```
