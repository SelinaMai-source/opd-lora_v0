# v1_seg_OPSD_replay：段内交替 + 80/20 replay

## 设置

- **方法**：与 `v1_seg_OPSD` 相同的 SFT/OPSD 交替与冻结 LoRA teacher；每个 epoch 的训练分布为 **约 80% 当前段 + 20% 已见段均匀抽样**。SFT 块与 OPSD 块共用该混合分布。
- **baseline_name**：`seg_opsd_replay`
- **replay**：`replay_ratio: 0.2`，`buffer_size: 2048`（够装 InstrDialog++ 全流 38×50），段训练结束后才把当前段写入 buffer（第一段无 replay）。实现复用 `replay_lora` 的 buffer / 均匀抽样思路。
- **K**：默认 25（与 B 的默认值对齐；B 网格胜出后再改 yaml）
- **epochs**：2（与 B 的 K=25 配置一致，保证一段内能跑完 SFT→OPSD）
- **初始化**：fresh LoRA
- **其余口径**：seed 123，r=16 / α=32 / dropout 0.05 / q,v_proj，lr 2e-4，bs 2，EOS mask false
- **实现**：`baselines/basic_baselines/seg_opsd/method.py` 的 `SegOPSDReplayMethod`

正式结果见 [results/README.md](results/README.md)。本轮最强变体：InstrDialog Score 0.378 / F 0.083，++ Score 0.379 / F 0.108，两条流均超过 v0 与无 replay 的 seg_OPSD。

## 复现（勿在 GPU 被占用时跑）

```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m core.train --config configs/smoke_v1_seg_opsd_replay.yaml
python -m core.train --config configs/v1_seg_opsd_replay_instrdialog.yaml
python -m core.train --config configs/v1_seg_opsd_replay_instrdialogpp.yaml
```
