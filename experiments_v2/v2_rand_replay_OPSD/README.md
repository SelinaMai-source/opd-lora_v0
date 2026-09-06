# v2_rand_replay_OPSD：同一步 CE+OPSD + 20% 随机 replay

## 设置

- **方法**：与 proto replay 同预算。总 replay 为当前段样本数的 20%，每段结束后该段只留 K=10，**均匀随机**抽样。λ=0.2。
- **baseline_name**：`ce_opsd_replay`
- **replay.strategy**：`uniform`
- **实现**：`baselines/basic_baselines/ce_opsd_replay/method.py`

对照：同预算 `v2_proto_replay` 与表内已有 `CITB_Replay K=10`（不重跑）。

## 复现

```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m core.train --config configs/v2_rand_replay_opsd_instrdialog_seed456.yaml
python -m core.train --config configs/v2_rand_replay_opsd_instrdialogpp_seed456.yaml
```

正式结果见 [results/README.md](results/README.md)。
