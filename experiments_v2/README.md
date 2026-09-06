# experiments_v2：同一步 CE+OPSD 家族（seed 456）

v2 接在「OPSD + 旧任务回放」上，训练默认是**同一步** `L = L_CE + λ L_OPSD`（不是 v1 `gold_opsd` 的 gold-token KL）。所有新 run 走 `benchmark/` 多参考流，`eval_normalization.format_normalize: true`，**seed 456**。不重跑 CITB / LCIA / v0 / v1。

## 方法

| 目录 | baseline_name | 网格 | 正式 job |
|---|---|---|---|
| [v2_multiref_gold_OPSD](v2_multiref_gold_OPSD/) | `ce_opsd` | λ∈{0.1,0.2,0.3} × 两流 | 6 |
| [v2_rand_replay_OPSD](v2_rand_replay_OPSD/) | `ce_opsd_replay` strategy=uniform | 20% 随机 replay, K=10 | 2 |
| [v2_proto_replay_OPSD](v2_proto_replay_OPSD/) | `ce_opsd_replay` strategy=proto | 10% 近邻 + 10% 退化 | 2 |
| [v2_proto_teacher_OPSD](v2_proto_teacher_OPSD/) | `ce_opsd_replay` + teacher_proto | random / matched × 两流 | 4（none = proto_replay，不重跑） |
| [v2_metric_OPSD](v2_metric_OPSD/) | `metric_opsd` | μ∈{0.05,0.1,0.2} × 两流 | 6 |

口径：Llama-3.1-8B-Instruct、LoRA r=16/α=32、lr 2e-4、bs 2、每段 1 epoch、`eval.debug_max_examples: 0`、`save_final_adapter: true`。

## 代码位置

运行时入口在仓库根目录的 `baselines/` / `core/`。本目录另有一份拷贝，便于 `v2` 分支自包含：

- `methods/ce_opsd/method.py`
- `methods/ce_opsd_replay/method.py`
- `methods/metric_opsd/method.py`
- teacher proto：`core/formatting.py` 的 `format_for_teacher(..., prototypes=)`
- 段末遗忘：`core/train.py` 在 `evaluate_stream` 之后调用 `on_eval_end`

数据：`scripts/build_v2_multiref_streams.py` → `benchmark/instrdialog(pp)/*_multiref.json`（不覆盖 CITB 旧 json）。

## 复现

```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 2 段冒烟通过后再串行 20 个正式 job；单卡 4090，勿开第二条 8B
/root/autodl-tmp/conda_envs/rp_lora_v0/bin/python scripts/run_v2_queue.py --mode all
```

正式结果齐后从各 `results/runs/<run_name>/final_metrics.json` 填飞书表。结果占位见各方法 `results/README.md`。
