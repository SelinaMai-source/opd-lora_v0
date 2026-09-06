# v2_proto_replay_OPSD：同一步 CE+OPSD + proto replay

## 设置

- **方法**：段末写 prototype（冻结 backbone 对 instruction 做 mean-pool；格式用 `task_score_type` + 金标均长；K=10 代表样本，尽量覆盖不同 input）。混合：10% 近邻（语义余弦高且格式兼容）+ 10% 退化（上一轮 eval `forgetting_by_segment` 最大的旧段）。第一段 100% 当前。λ=0.2。
- **baseline_name**：`ce_opsd_replay`
- **replay.strategy**：`proto`
- **teacher_proto.mode**：`none`（与 teacher 三组中的 none 同一 job，不重跑）
- **实现**：`baselines/basic_baselines/ce_opsd_replay/method.py`；`core/train.py` 段末 `on_eval_end`

## 复现

```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m core.train --config configs/v2_proto_replay_opsd_instrdialog_seed456_smoke2seg.yaml
python -m core.train --config configs/v2_proto_replay_opsd_instrdialog_seed456.yaml
python -m core.train --config configs/v2_proto_replay_opsd_instrdialogpp_seed456.yaml
```

正式结果见 [results/README.md](results/README.md)。
