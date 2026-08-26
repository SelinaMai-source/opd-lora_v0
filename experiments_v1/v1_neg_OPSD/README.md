# v1_neg_OPSD：模板负样本 pairwise（叠在 SFT CE 上）

## 设置

- **方法**：对 gold 做模板扩写负样本：`The answer is {gold}` / `The final answer is {gold}` / `Sure, {gold}`。每 batch `L = L_CE + μ L_pref`，其中 `L_pref` 为 gold vs 每个 neg 的 Bradley-Terry / DPO 风格 pairwise：`-log σ(β (logp_gold − logp_neg))`，对三个模板取均值。logp 为监督 span 上的 **token 均值**（长度归一化，避免负样本因前缀更长而天然 logp 更低）。
- **baseline_name**：`neg_opsd`
- **μ**：默认 `0.3`；**β**：默认 `1.0`（长度归一化后的 BT 温度，可在 `neg_opsd.beta` 改）
- **初始化**：fresh LoRA（叠在 sequential SFT 上，不加载 SFT adapter，也不叠加蒸馏）
- **其余口径**：seed 123，r=16 / α=32 / dropout 0.05 / q,v_proj，lr 2e-4，bs 2，每段 1 epoch，EOS mask false
- **实现**：`baselines/basic_baselines/neg_opsd/method.py`
- **长序列**：监督序列 ≥1024 token 时跳过 pairwise 只留 CE；logp 用 fused CE nll，避免 vocab 维 logsumexp OOM（CNN/DM）

正式结果见 [results/README.md](results/README.md)。InstrDialog Score 0.325 / F 0.100（略低于 v0）；InstrDialog++ Score 0.337 / F 0.113（超过 v0 的 0.285 / 0.180）。seg 27 CNN/DM 长序列已用 fused CE nll + 超长跳过 pairwise 跑通。

## 复现（勿在 GPU 被占用时跑）

```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m core.train --config configs/smoke_v1_neg_opsd.yaml
python -m core.train --config configs/v1_neg_opsd_instrdialog.yaml
python -m core.train --config configs/v1_neg_opsd_instrdialogpp.yaml
```
