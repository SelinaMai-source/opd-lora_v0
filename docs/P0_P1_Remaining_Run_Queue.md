# P0 / P1 Remaining Run Queue

Protocol B 三种子 2×2。8 个 seed 456 已 Verified Completed，不重跑。
16 个新 run 的 `holdout_seed` 固定 456，只改训练 `seed` 与 `output.run_name`。
训练入口：`/root/autodl-tmp/conda_envs/rp_lora_v0/bin/python -m core.trainer --config <yaml>`。
结果根：`/root/autodl-tmp/rp_lora_v3_runs`。单进程，锁 `/tmp/rp_lora_v3_gpu.lock`。
指纹一致则跳过；目录已在且不完整则拒绝覆盖。不启动 E2–E12。

| Run ID | Benchmark | Method | Seed | Protocol | Status | Config | Result Path | GPU Time | Notes |
|---|---|---|---|---|---|---|---|---|---|
| v3h_rand_ce_instrdialog_seed456 | instrdialog | rand_ce | 456 | Protocol B train_holdout | Verified Completed | configs/v3h_rand_ce_instrdialog_seed456.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_ce_instrdialog_seed456 | 623.5s | 复用，不重跑；uniform 回放 9；无监控前向选样；holdout_seed=456 |
| v3h_proto_ce_instrdialog_seed456 | instrdialog | proto_ce | 456 | Protocol B train_holdout | Verified Completed | configs/v3h_proto_ce_instrdialog_seed456.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_ce_instrdialog_seed456 | 643.8s | 复用，不重跑；PGDR 每段约 5 条监控前向；proto 回放 8（4+4）；holdout_seed=456 |
| v3h_rand_opsd_instrdialog_seed456 | instrdialog | rand_opsd | 456 | Protocol B train_holdout | Verified Completed | configs/v3h_rand_opsd_instrdialog_seed456.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_opsd_instrdialog_seed456 | 1257.5s | 复用，不重跑；uniform 回放 9；无监控前向选样；holdout_seed=456 |
| v3h_proto_opsd_instrdialog_seed456 | instrdialog | proto_opsd | 456 | Protocol B train_holdout | Verified Completed | configs/v3h_proto_opsd_instrdialog_seed456.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_opsd_instrdialog_seed456 | 1246.3s | 复用，不重跑；PGDR 每段约 5 条监控前向；proto 回放 8（4+4）；holdout_seed=456 |
| v3h_rand_ce_instrdialogpp_seed456 | instrdialogpp | rand_ce | 456 | Protocol B train_holdout | Verified Completed | configs/v3h_rand_ce_instrdialogpp_seed456.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_ce_instrdialogpp_seed456 | 1893.5s | 复用，不重跑；uniform 回放 9；无监控前向选样；holdout_seed=456 |
| v3h_proto_ce_instrdialogpp_seed456 | instrdialogpp | proto_ce | 456 | Protocol B train_holdout | Verified Completed | configs/v3h_proto_ce_instrdialogpp_seed456.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_ce_instrdialogpp_seed456 | 1829.3s | 复用，不重跑；PGDR 每段约 5 条监控前向；proto 回放 8（4+4）；holdout_seed=456 |
| v3h_rand_opsd_instrdialogpp_seed456 | instrdialogpp | rand_opsd | 456 | Protocol B train_holdout | Verified Completed | configs/v3h_rand_opsd_instrdialogpp_seed456.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_opsd_instrdialogpp_seed456 | 3109.5s | 复用，不重跑；uniform 回放 9；无监控前向选样；holdout_seed=456 |
| v3h_proto_opsd_instrdialogpp_seed456 | instrdialogpp | proto_opsd | 456 | Protocol B train_holdout | Verified Completed | configs/v3h_proto_opsd_instrdialogpp_seed456.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_opsd_instrdialogpp_seed456 | 2951.2s | 复用，不重跑；PGDR 每段约 5 条监控前向；proto 回放 8（4+4）；holdout_seed=456 |
| v3h_proto_ce_instrdialog_seed789 | instrdialog | proto_ce | 789 | Protocol B train_holdout | Completed | configs/v3h_proto_ce_instrdialog_seed789.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_ce_instrdialog_seed789 | 618.9s | 新跑完；PGDR 每段约 5 条监控前向；proto 回放 8（4+4）；holdout_seed=456 |
| v3h_proto_ce_instrdialog_seed42 | instrdialog | proto_ce | 42 | Protocol B train_holdout | Completed | configs/v3h_proto_ce_instrdialog_seed42.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_ce_instrdialog_seed42 | 584.4s | 新跑完；PGDR 每段约 5 条监控前向；proto 回放 8（4+4）；holdout_seed=456 |
| v3h_proto_opsd_instrdialog_seed789 | instrdialog | proto_opsd | 789 | Protocol B train_holdout | Completed | configs/v3h_proto_opsd_instrdialog_seed789.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_opsd_instrdialog_seed789 | 1197.7s | 新跑完；PGDR 每段约 5 条监控前向；proto 回放 8（4+4）；holdout_seed=456 |
| v3h_proto_opsd_instrdialog_seed42 | instrdialog | proto_opsd | 42 | Protocol B train_holdout | Completed | configs/v3h_proto_opsd_instrdialog_seed42.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_opsd_instrdialog_seed42 | 1183.1s | 新跑完；PGDR 每段约 5 条监控前向；proto 回放 8（4+4）；holdout_seed=456 |
| v3h_proto_ce_instrdialogpp_seed789 | instrdialogpp | proto_ce | 789 | Protocol B train_holdout | Completed | configs/v3h_proto_ce_instrdialogpp_seed789.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_ce_instrdialogpp_seed789 | 1810.0s | 新跑完；PGDR 每段约 5 条监控前向；proto 回放 8（4+4）；holdout_seed=456 |
| v3h_proto_ce_instrdialogpp_seed42 | instrdialogpp | proto_ce | 42 | Protocol B train_holdout | Completed | configs/v3h_proto_ce_instrdialogpp_seed42.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_ce_instrdialogpp_seed42 | 1801.3s | 新跑完；PGDR 每段约 5 条监控前向；proto 回放 8（4+4）；holdout_seed=456 |
| v3h_proto_opsd_instrdialogpp_seed789 | instrdialogpp | proto_opsd | 789 | Protocol B train_holdout | Completed | configs/v3h_proto_opsd_instrdialogpp_seed789.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_opsd_instrdialogpp_seed789 | 2872.3s | 新跑完；PGDR 每段约 5 条监控前向；proto 回放 8（4+4）；holdout_seed=456 |
| v3h_proto_opsd_instrdialogpp_seed42 | instrdialogpp | proto_opsd | 42 | Protocol B train_holdout | Completed | configs/v3h_proto_opsd_instrdialogpp_seed42.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_opsd_instrdialogpp_seed42 | 2801.8s | 新跑完；PGDR 每段约 5 条监控前向；proto 回放 8（4+4）；holdout_seed=456 |
| v3h_rand_ce_instrdialog_seed789 | instrdialog | rand_ce | 789 | Protocol B train_holdout | Completed | configs/v3h_rand_ce_instrdialog_seed789.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_ce_instrdialog_seed789 | 596.4s | 新跑完；uniform 回放 9；无监控前向选样；holdout_seed=456 |
| v3h_rand_ce_instrdialog_seed42 | instrdialog | rand_ce | 42 | Protocol B train_holdout | Completed | configs/v3h_rand_ce_instrdialog_seed42.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_ce_instrdialog_seed42 | 635.9s | 新跑完；uniform 回放 9；无监控前向选样；holdout_seed=456 |
| v3h_rand_opsd_instrdialog_seed789 | instrdialog | rand_opsd | 789 | Protocol B train_holdout | Completed | configs/v3h_rand_opsd_instrdialog_seed789.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_opsd_instrdialog_seed789 | 1184.1s | 新跑完；uniform 回放 9；无监控前向选样；holdout_seed=456 |
| v3h_rand_opsd_instrdialog_seed42 | instrdialog | rand_opsd | 42 | Protocol B train_holdout | Completed | configs/v3h_rand_opsd_instrdialog_seed42.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_opsd_instrdialog_seed42 | 1223.6s | 新跑完；uniform 回放 9；无监控前向选样；holdout_seed=456 |
| v3h_rand_ce_instrdialogpp_seed789 | instrdialogpp | rand_ce | 789 | Protocol B train_holdout | Completed | configs/v3h_rand_ce_instrdialogpp_seed789.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_ce_instrdialogpp_seed789 | 1873.8s | 新跑完；uniform 回放 9；无监控前向选样；holdout_seed=456 |
| v3h_rand_ce_instrdialogpp_seed42 | instrdialogpp | rand_ce | 42 | Protocol B train_holdout | Completed | configs/v3h_rand_ce_instrdialogpp_seed42.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_ce_instrdialogpp_seed42 | 1907.2s | 新跑完；uniform 回放 9；无监控前向选样；holdout_seed=456 |
| v3h_rand_opsd_instrdialogpp_seed789 | instrdialogpp | rand_opsd | 789 | Protocol B train_holdout | Completed | configs/v3h_rand_opsd_instrdialogpp_seed789.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_opsd_instrdialogpp_seed789 | 3119.4s | 新跑完；uniform 回放 9；无监控前向选样；holdout_seed=456 |
| v3h_rand_opsd_instrdialogpp_seed42 | instrdialogpp | rand_opsd | 42 | Protocol B train_holdout | Completed | configs/v3h_rand_opsd_instrdialogpp_seed42.yaml | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_opsd_instrdialogpp_seed42 | 3119.5s | 新跑完；uniform 回放 9；无监控前向选样；holdout_seed=456 |

## Resume

中断后不要覆盖已有目录。原样续跑整队：

```bash
cd /root/research-repos/opd-lora_v3
export PYTHONPATH=/root/research-repos/opd-lora_v3
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TORCH_HOME=/root/autodl-tmp/torch_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/root/autodl-tmp/conda_envs/rp_lora_v0/bin/python scripts/run_v3h_p0_3seed_queue.py
```

单条续跑命令在 `/root/autodl-tmp/rp_lora_v3_runs/logs/<run>.resume_cmd`。
