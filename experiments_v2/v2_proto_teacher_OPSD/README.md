# v2_proto_teacher_OPSD：teacher 加旧任务 prototype

## 设置

- **方法**：同一 proto replay 骨干（λ=0.2）。student 仍只看当前 instruction+input。teacher 在 privileged gold 之外再塞 1～2 条旧任务摘要+例子（`format_for_teacher(..., prototypes=)`）。
- **三组**：`none`（= `v2_proto_replay`，不重跑）/ `random` / `matched`
- **baseline_name**：`ce_opsd_replay`
- **teacher_proto.mode**：`random` 或 `matched`
- **实现**：`core/formatting.py`、`baselines/basic_baselines/ce_opsd_replay/method.py`

重点看 matched 是否用更少更新学会新任务（段内 `current_score` 与 `train.n_steps`）。

## 复现

```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m core.train --config configs/v2_proto_teacher_opsd_random_instrdialog_seed456.yaml
python -m core.train --config configs/v2_proto_teacher_opsd_matched_instrdialog_seed456.yaml
```

正式结果见 [results/README.md](results/README.md)。
