# v1_SFT_OPSD：SFT 初始化 + OPSD

## 设置

- **方法**：与 v1_OPSD 相同的 OPSD 训练目标（on-policy rollout + 双 forward + 逐 token JSD β=0.5），唯一差异是**首个 segment 前通过 `model.init_adapter_path` 加载 v0 SFT 重跑保存的最终 adapter**，在 SFT 基础上继续做 OPSD。
- **SFT adapter 来源**：`results/runs/v0_sft_instrdialog/final_adapter`、`results/runs/v0_sft_instrdialogpp/final_adapter`（不入库，需按下方命令先重跑 SFT）
- **其余口径**：与 v0/v1_OPSD 完全一致（r=16/α=32/q,v_proj/lr 2e-4/bs 2/每段 1 epoch/seed 123，Llama-3.1-8B-Instruct bfloat16）
- **配置**：`configs/` 内为本变体两条流的 yaml 副本

## 结果（全指标，提取自各 run 的 final_metrics.json，数值保留 3 位小数）

| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | Current-Seg Acc↑ | Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog | 0.346 | 0.362 | 0.089 | 0.083 | 0.430 | 0.425 | 0.451 | 0.422 | 0.000 | 0.000 | 0.332 | 合理：SFT 基线，19 段流遗忘低（F 0.089），生成质量指标为全表最高档，作为 v1 对照起点 |
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog++ | 0.285 | 0.298 | 0.180 | 0.175 | 0.392 | 0.388 | 0.461 | 0.383 | 0.100 | 0.100 | 0.261 | 合理：38 段长流遗忘明显加重（F 0.180 vs 19 段 0.089），符合流越长越难抗遗忘的预期 |
| v1_OPSD | fresh LoRA 直接 OPSD，无 SFT 初始化 | InstrDialog | 0.289 | 0.299 | 0.128 | 0.133 | 0.361 | 0.357 | 0.385 | 0.357 | 0.000 | 0.000 | 0.270 | 合理但偏弱：训练稳定无崩盘、F 与 v0 相当，但 Score 低于 SFT 约 5.8pp——仅 on-policy 蒸馏不足以替代 SFT 起点 |
| v1_OPSD | fresh LoRA 直接 OPSD，无 SFT 初始化 | InstrDialog++ | 0.223 | 0.255 | 0.171 | 0.154 | 0.359 | 0.352 | 0.423 | 0.370 | 0.100 | 0.300 | 0.224 | 合理但偏弱：同样稳定无崩盘，Score 低于 v0 约 6.2pp、F 与 v0 接近，结论与 19 段流一致 |
| v1_SFT_OPSD | 加载 SFT 最终 adapter 初始化，继续 OPSD（原 run seed 123） | InstrDialog | 0.011 | 0.211 | 0.367 | 0.171 | 0.153 | 0.146 | 0.197 | 0.165 | 0.000 | 0.000 | 0.190 | 异常（单次）：末段（segment 18, task1600_smcalflow）训练后 strict EM 由段 17 末的 0.294 崩至 0.011（F 0.367）；2026-08-24 同 seed 重跑未崩、seed 456 亦未崩 → GPU 非确定性 + seed 敏感临界点，非该流系统性失稳 |
| v1_SFT_OPSD | 同 seed 123 重跑（每段保存 adapter） | InstrDialog | 0.263 | 0.283 | 0.182 | 0.172 | 0.361 | 0.353 | 0.396 | 0.366 | 0.000 | 0.000 | 0.256 | 未崩：段 18 后 Score 0.263（相对段 17 的 0.210 回升），无法复现原 0.011 崩盘 |
| v1_SFT_OPSD | seed 456 稳健性（同 SFT init adapter） | InstrDialog | 0.347 | 0.358 | 0.033 | 0.033 | 0.420 | 0.415 | 0.443 | 0.423 | 0.000 | 0.000 | 0.322 | 未崩且最优档：Score 0.347 与 v0 SFT 持平，F 0.033 为 InstrDialog 全表最低 |
| v1_SFT_OPSD | 加载 SFT 最终 adapter 初始化，继续 OPSD | InstrDialog++ | 0.300 | 0.313 | 0.084 | 0.082 | 0.409 | 0.402 | 0.457 | 0.412 | 0.000 | 0.200 | 0.272 | 合理且最优：Score 0.300 为长流最高，F 0.084 显著优于 v0 的 0.180——SFT 初始化 + OPSD 在长流上收益明确 |

注：Seen-Avg Acc = `eval.seen_avg_score`（strict EM），Forgetting = `eval.forgetting`，Current-Seg Acc = `eval.current_score`（末段），Task-aware Score Mean = `eval.task_aware_score_mean`（训练轨迹均值）；Output Length 指标已从表中移除，崩盘行输出失控现象见结果解释。完整口径见 `docs/v1_experiment_report.md`（/root/autodl-tmp/docs/）。

## 末段崩盘诊断（InstrDialog 流）与重跑

- 原 run `20260820_181411_v1_sft_opsd_instrdialog`：segment 17 结束时 seen_avg **0.2944**（F 0.088），segment 18（`task1600_smcalflow_sentence_generation`）后崩至 **0.0105**（F 0.367）。
- 训练侧指标正常（末段 JSD loss 0.117、grad_norm 1.02），同段 v0/v1_OPSD 不崩 → **排除实现 bug 与数据段异常**。
- **2026-08-24 重跑**：同 seed 123 → Score **0.263** / F 0.182（段 17→18 由 0.210 回升，未崩）；seed 456 → Score **0.347** / F 0.033（未崩）。判定 **C2+C4**：含 GPU 非确定性，且为 seed 敏感临界点，**非系统性失稳**。
- 段级 adapter：seed123/456 的 17→18 相对 L2 分别为 0.080 / 0.066，均小于 16→17，健康轨迹下无异常大权重跳变。

## run id 与产物

| 流 | run id | 配置 |
|---|---|---|
| InstrDialog（19 段，原 run） | `20260820_181411_v1_sft_opsd_instrdialog` | v1_sft_opsd_instrdialog.yaml |
| InstrDialog（seed 123 重跑） | `v1_sft_opsd_instrdialog_seed123_rerun` | v1_sft_opsd_instrdialog_rerun_seed123.yaml |
| InstrDialog（seed 456） | `v1_sft_opsd_instrdialog_seed456` | v1_sft_opsd_instrdialog_seed456.yaml |
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
# InstrDialog 重跑验证
python -m core.train --config configs/v1_sft_opsd_instrdialog_rerun_seed123.yaml
python -m core.train --config configs/v1_sft_opsd_instrdialog_seed456.yaml
```
