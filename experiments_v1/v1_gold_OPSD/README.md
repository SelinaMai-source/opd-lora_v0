# v1_gold_OPSD：同一步 CE + gold 上 KL 蒸馏

## 设置

- **方法**：每 batch `L = L_CE + λ L_KL`。Student 在 gold continuation 上做标准 SFT CE；Teacher 用 `format_for_teacher` + `no_grad` **冻结基座**（`disable_adapter`）；KL(p_T ∥ p_S) 打在**同一 gold token 序列**上（off-policy 蒸馏，不是 on-policy rollout）。只更新 student LoRA。
- **baseline_name**：`gold_opsd`
- **λ**：默认 `0.3`（`gold_opsd.lambda_kl`）
- **格式约束**：开启（`opsd.teacher_format_constraint: true`）
- **初始化**：fresh LoRA（每步已含 CE，不加载 SFT adapter）
- **其余口径**：seed 123，r=16 / α=32 / dropout 0.05 / q,v_proj，lr 2e-4，bs 2，每段 1 epoch，EOS mask false
- **实现**：`baselines/basic_baselines/gold_opsd/method.py`

正式结果见 [results/README.md](results/README.md)。与 v0 SFT 接近（InstrDialog 0.341 vs 0.346；++ 0.303 vs 0.285），符合主项仍是 CE 的设计。

## 复现（勿在 GPU 被占用时跑）

```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m core.train --config configs/smoke_v1_gold_opsd.yaml
python -m core.train --config configs/v1_gold_opsd_instrdialog.yaml
python -m core.train --config configs/v1_gold_opsd_instrdialogpp.yaml
```
