# v1_OPSD：fresh LoRA 直接 OPSD（无 SFT 初始化）

## 设置

- **方法**：OPSD（on-policy self-distillation）。每 batch：当前 student 采样 rollout（temperature=1.0）→ 同一 rollout 接在 student prompt 与 teacher prompt（含 privileged reference `y*` + transition instruction）后做双 forward → rollout 逐 token generalized JSD（β=0.5）→ 只更新 student adapter。Teacher 为冻结基座（PEFT disable_adapter，不占双份显存）。
- **初始化**：fresh LoRA（r=16, alpha=32, dropout=0.05, q/v_proj, lr 2e-4, bs 2, 每段 1 epoch, seed 123）
- **Base Model**：Meta-Llama-3.1-8B-Instruct（bfloat16）
- **实现代码**：`baselines/basic_baselines/opsd/method.py`、`core/formatting.py`（format_for_teacher）、`core/models/base_model.py`（sample_rollouts / forward_logits）
- **配置**：`configs/` 内为本变体两条流的 yaml 副本（与主 `configs/` 一致）

## 结果（Score = seen_avg strict EM，F = forgetting）

| 方法 | Base Model | LoRA 设置 | 评测结果(InstrDialog: Score/F, InstrDialog++: Score/F) | 关键发现 |
|---|---|---|---|---|
| v1_OPSD | Llama-3.1-8B-Instruct | r=16/α=32/q,v_proj/lr 2e-4，fresh LoRA（无 SFT 初始化） | 28.86% / 0.128；22.31% / 0.171 | 两条流均稳定训练、无崩盘，F 与 v0 SFT 相当；但绝对 Score 低于 SFT 系 5.8/6.2 个百分点——仅 on-policy 蒸馏不足以替代 SFT 起点 |

参考（task-aware）：InstrDialog 0.2991 / F 0.133；InstrDialog++ 0.2547 / F 0.154。

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
