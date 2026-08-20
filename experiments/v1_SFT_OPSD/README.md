# v1_SFT_OPSD：SFT 初始化 + OPSD

## 设置

- **方法**：与 v1_OPSD 相同的 OPSD 训练目标（on-policy rollout + 双 forward + 逐 token JSD β=0.5），唯一差异是**首个 segment 前通过 `model.init_adapter_path` 加载 v0 SFT 重跑保存的最终 adapter**，在 SFT 基础上继续做 OPSD。
- **SFT adapter 来源**：`results/runs/v0_sft_instrdialog/final_adapter`、`results/runs/v0_sft_instrdialogpp/final_adapter`（不入库，需按下方命令先重跑 SFT）
- **其余口径**：与 v0/v1_OPSD 完全一致（r=16/α=32/q,v_proj/lr 2e-4/bs 2/每段 1 epoch/seed 123，Llama-3.1-8B-Instruct bfloat16）
- **配置**：`configs/` 内为本变体两条流的 yaml 副本

## 结果（Score = seen_avg strict EM，F = forgetting）

| 方法 | Base Model | LoRA 设置 | 评测结果(InstrDialog: Score/F, InstrDialog++: Score/F) | 关键发现 |
|---|---|---|---|---|
| v1_SFT_OPSD | Llama-3.1-8B-Instruct | r=16/α=32/q,v_proj/lr 2e-4，加载 SFT 最终 adapter 初始化 | 1.05% / 0.367；30.03% / 0.084 | InstrDialog++ 上 Score 30.03% 超 v0 SFT（28.50%）且 F 0.084 显著优于 v0 的 0.180；但 InstrDialog 末段（segment 18, task1600_smcalflow）训练后 strict EM 崩至 1.05%（F 0.367），经诊断排除 bug、属真实剧烈遗忘——SFT 初始化对 OPSD 是双刃剑 |

参考（task-aware）：InstrDialog 0.2105 / F 0.171；InstrDialog++ 0.3135 / F 0.082。

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
