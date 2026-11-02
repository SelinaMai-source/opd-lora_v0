# ACL v3 Factorial 3-Seed Full Results

Protocol B only（`monitor: train_holdout`，`holdout_seed: 456`，`eval_normalization.format_normalize: true`，`benchmark/*_multiref.json`）。版本列统一 `v3_factorial_heldout_3seed`。24 行 = 4 方法 × 2 数据 × 3 seed。缺 `final_metrics.json` 的格写 —，不估数。

复用 8 个已验证 seed 456；本文件不把 Protocol A 行写进主表。科学限制保持：每段 fit 45 / 监控 5；uniform 回放 9、proto 8；近邻槽不是余弦检索。

已测 24 / 24。

## Protocol B 24 行

| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | Current-Seg Acc↑ | Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3_factorial_heldout_3seed | 随机replay+CE,K=10/段,20% replay,λ=0,train_holdout holdout_seed=456, seed 456 | InstrDialog | 0.321 | 0.326 | 0.128 | 0.133 | 0.408 | 0.404 | 0.450 | 0.398 | 0.000 | 0.000 | 0.326 | 合理:v3h_rand_ce_instrdialog_seed456;Acc 0.321,F 0.128;同seed配对 proto_ce-本格 Acc +0.042,F -0.044;同seed配对 rand_opsd-本格 Acc +0.074,F -0.050;来源 final.eval.* |
| v3_factorial_heldout_3seed | 随机replay+CE,K=10/段,20% replay,λ=0,train_holdout holdout_seed=456, seed 789 | InstrDialog | 0.305 | 0.311 | 0.139 | 0.139 | 0.406 | 0.399 | 0.448 | 0.397 | 0.000 | 0.000 | 0.311 | 合理:v3h_rand_ce_instrdialog_seed789;Acc 0.305,F 0.139;同seed配对 proto_ce-本格 Acc +0.095,F -0.083;同seed配对 rand_opsd-本格 Acc +0.063,F -0.061;来源 final.eval.* |
| v3_factorial_heldout_3seed | 随机replay+CE,K=10/段,20% replay,λ=0,train_holdout holdout_seed=456, seed 42 | InstrDialog | 0.368 | 0.384 | 0.089 | 0.083 | 0.490 | 0.484 | 0.491 | 0.480 | 0.000 | 0.000 | 0.384 | 合理:v3h_rand_ce_instrdialog_seed42;Acc 0.368,F 0.089;同seed配对 proto_ce-本格 Acc -0.042,F +0.028;同seed配对 rand_opsd-本格 Acc -0.026,F -0.006;来源 final.eval.* |
| v3_factorial_heldout_3seed | prototype replay+CE,K=10/段,20% proto replay(10%近邻/10%退化),λ=0,train_holdout holdout_seed=456, seed 456 | InstrDialog | 0.363 | 0.379 | 0.083 | 0.072 | 0.455 | 0.451 | 0.490 | 0.442 | 0.000 | 0.000 | 0.379 | 合理:v3h_proto_ce_instrdialog_seed456;Acc 0.363,F 0.083;同seed配对 vs rand_ce Acc +0.042,F -0.044;同seed配对 proto_opsd-本格 Acc +0.021,F +0.006;来源 final.eval.* |
| v3_factorial_heldout_3seed | prototype replay+CE,K=10/段,20% proto replay(10%近邻/10%退化),λ=0,train_holdout holdout_seed=456, seed 789 | InstrDialog | 0.400 | 0.416 | 0.056 | 0.039 | 0.496 | 0.491 | 0.517 | 0.487 | 0.000 | 0.000 | 0.416 | 合理:v3h_proto_ce_instrdialog_seed789;Acc 0.400,F 0.056;同seed配对 vs rand_ce Acc +0.095,F -0.083;同seed配对 proto_opsd-本格 Acc -0.016,F +0.006;来源 final.eval.* |
| v3_factorial_heldout_3seed | prototype replay+CE,K=10/段,20% proto replay(10%近邻/10%退化),λ=0,train_holdout holdout_seed=456, seed 42 | InstrDialog | 0.326 | 0.337 | 0.117 | 0.111 | 0.431 | 0.426 | 0.479 | 0.420 | 0.000 | 0.000 | 0.337 | 负向:v3h_proto_ce_instrdialog_seed42;Acc 0.326,F 0.117;同seed配对 vs rand_ce Acc -0.042,F +0.028;同seed配对 proto_opsd-本格 Acc +0.058,F -0.050;来源 final.eval.* |
| v3_factorial_heldout_3seed | 随机replay+CE+λ=0.2 OPSD,K=10/段,20% replay,train_holdout holdout_seed=456, seed 456 | InstrDialog | 0.395 | 0.395 | 0.078 | 0.078 | 0.498 | 0.493 | 0.513 | 0.491 | 0.000 | 0.000 | 0.395 | 合理:v3h_rand_opsd_instrdialog_seed456;Acc 0.395,F 0.078;同seed配对 vs rand_ce Acc +0.074,F -0.050;同seed配对 proto_opsd-本格 Acc -0.011,F +0.011;来源 final.eval.* |
| v3_factorial_heldout_3seed | 随机replay+CE+λ=0.2 OPSD,K=10/段,20% replay,train_holdout holdout_seed=456, seed 789 | InstrDialog | 0.368 | 0.384 | 0.078 | 0.072 | 0.480 | 0.476 | 0.501 | 0.475 | 0.000 | 0.000 | 0.384 | 合理:v3h_rand_opsd_instrdialog_seed789;Acc 0.368,F 0.078;同seed配对 vs rand_ce Acc +0.063,F -0.061;同seed配对 proto_opsd-本格 Acc +0.016,F -0.017;来源 final.eval.* |
| v3_factorial_heldout_3seed | 随机replay+CE+λ=0.2 OPSD,K=10/段,20% replay,train_holdout holdout_seed=456, seed 42 | InstrDialog | 0.342 | 0.358 | 0.083 | 0.078 | 0.466 | 0.460 | 0.477 | 0.468 | 0.000 | 0.000 | 0.358 | 负向:v3h_rand_opsd_instrdialog_seed42;Acc 0.342,F 0.083;同seed配对 vs rand_ce Acc -0.026,F -0.006;同seed配对 proto_opsd-本格 Acc +0.042,F -0.017;来源 final.eval.* |
| v3_factorial_heldout_3seed | prototype replay+CE+λ=0.2 OPSD,K=10/段,20% proto replay(10%近邻/10%退化),train_holdout holdout_seed=456, seed 456 | InstrDialog | 0.384 | 0.389 | 0.089 | 0.083 | 0.466 | 0.460 | 0.484 | 0.452 | 0.000 | 0.000 | 0.389 | 合理:v3h_proto_opsd_instrdialog_seed456;Acc 0.384,F 0.089;同seed配对 vs proto_ce Acc +0.021,F +0.006;同seed配对 vs rand_opsd Acc -0.011,F +0.011;来源 final.eval.* |
| v3_factorial_heldout_3seed | prototype replay+CE+λ=0.2 OPSD,K=10/段,20% proto replay(10%近邻/10%退化),train_holdout holdout_seed=456, seed 789 | InstrDialog | 0.384 | 0.395 | 0.061 | 0.050 | 0.495 | 0.490 | 0.510 | 0.488 | 0.000 | 0.000 | 0.395 | 合理:v3h_proto_opsd_instrdialog_seed789;Acc 0.384,F 0.061;同seed配对 vs proto_ce Acc -0.016,F +0.006;同seed配对 vs rand_opsd Acc +0.016,F -0.017;来源 final.eval.* |
| v3_factorial_heldout_3seed | prototype replay+CE+λ=0.2 OPSD,K=10/段,20% proto replay(10%近邻/10%退化),train_holdout holdout_seed=456, seed 42 | InstrDialog | 0.384 | 0.400 | 0.067 | 0.056 | 0.491 | 0.486 | 0.505 | 0.483 | 0.000 | 0.000 | 0.400 | 合理:v3h_proto_opsd_instrdialog_seed42;Acc 0.384,F 0.067;同seed配对 vs proto_ce Acc +0.058,F -0.050;同seed配对 vs rand_opsd Acc +0.042,F -0.017;来源 final.eval.* |
| v3_factorial_heldout_3seed | 随机replay+CE,K=10/段,20% replay,λ=0,train_holdout holdout_seed=456, seed 456 | InstrDialog++ | 0.411 | 0.411 | 0.124 | 0.127 | 0.543 | 0.535 | 0.580 | 0.534 | 0.200 | 0.200 | 0.411 | 合理:v3h_rand_ce_instrdialogpp_seed456;Acc 0.411,F 0.124;同seed配对 proto_ce-本格 Acc +0.016,F -0.030;同seed配对 rand_opsd-本格 Acc +0.005,F -0.022;来源 final.eval.* |
| v3_factorial_heldout_3seed | 随机replay+CE,K=10/段,20% replay,λ=0,train_holdout holdout_seed=456, seed 789 | InstrDialog++ | 0.358 | 0.358 | 0.173 | 0.176 | 0.488 | 0.483 | 0.546 | 0.476 | 0.000 | 0.000 | 0.358 | 合理:v3h_rand_ce_instrdialogpp_seed789;Acc 0.358,F 0.173;同seed配对 proto_ce-本格 Acc +0.074,F -0.105;同seed配对 rand_opsd-本格 Acc +0.066,F -0.078;来源 final.eval.* |
| v3_factorial_heldout_3seed | 随机replay+CE,K=10/段,20% replay,λ=0,train_holdout holdout_seed=456, seed 42 | InstrDialog++ | 0.395 | 0.408 | 0.127 | 0.116 | 0.505 | 0.500 | 0.554 | 0.501 | 0.200 | 0.200 | 0.408 | 合理:v3h_rand_ce_instrdialogpp_seed42;Acc 0.395,F 0.127;同seed配对 proto_ce-本格 Acc +0.024,F -0.057;同seed配对 rand_opsd-本格 Acc +0.045,F -0.057;来源 final.eval.* |
| v3_factorial_heldout_3seed | prototype replay+CE,K=10/段,20% proto replay(10%近邻/10%退化),λ=0,train_holdout holdout_seed=456, seed 456 | InstrDialog++ | 0.426 | 0.429 | 0.095 | 0.095 | 0.537 | 0.529 | 0.569 | 0.525 | 0.000 | 0.000 | 0.429 | 合理:v3h_proto_ce_instrdialogpp_seed456;Acc 0.426,F 0.095;同seed配对 vs rand_ce Acc +0.016,F -0.030;同seed配对 proto_opsd-本格 Acc +0.034,F -0.032;来源 final.eval.* |
| v3_factorial_heldout_3seed | prototype replay+CE,K=10/段,20% proto replay(10%近邻/10%退化),λ=0,train_holdout holdout_seed=456, seed 789 | InstrDialog++ | 0.432 | 0.432 | 0.068 | 0.068 | 0.533 | 0.525 | 0.582 | 0.523 | 0.000 | 0.000 | 0.432 | 合理:v3h_proto_ce_instrdialogpp_seed789;Acc 0.432,F 0.068;同seed配对 vs rand_ce Acc +0.074,F -0.105;同seed配对 proto_opsd-本格 Acc -0.032,F +0.030;来源 final.eval.* |
| v3_factorial_heldout_3seed | prototype replay+CE,K=10/段,20% proto replay(10%近邻/10%退化),λ=0,train_holdout holdout_seed=456, seed 42 | InstrDialog++ | 0.418 | 0.421 | 0.070 | 0.070 | 0.542 | 0.536 | 0.579 | 0.531 | 0.000 | 0.000 | 0.421 | 合理:v3h_proto_ce_instrdialogpp_seed42;Acc 0.418,F 0.070;同seed配对 vs rand_ce Acc +0.024,F -0.057;同seed配对 proto_opsd-本格 Acc +0.008,F +0.014;来源 final.eval.* |
| v3_factorial_heldout_3seed | 随机replay+CE+λ=0.2 OPSD,K=10/段,20% replay,train_holdout holdout_seed=456, seed 456 | InstrDialog++ | 0.416 | 0.421 | 0.103 | 0.103 | 0.540 | 0.532 | 0.581 | 0.529 | 0.200 | 0.200 | 0.421 | 合理:v3h_rand_opsd_instrdialogpp_seed456;Acc 0.416,F 0.103;同seed配对 vs rand_ce Acc +0.005,F -0.022;同seed配对 proto_opsd-本格 Acc +0.045,F -0.041;来源 final.eval.* |
| v3_factorial_heldout_3seed | 随机replay+CE+λ=0.2 OPSD,K=10/段,20% replay,train_holdout holdout_seed=456, seed 789 | InstrDialog++ | 0.424 | 0.426 | 0.095 | 0.095 | 0.558 | 0.551 | 0.586 | 0.545 | 0.000 | 0.000 | 0.426 | 合理:v3h_rand_opsd_instrdialogpp_seed789;Acc 0.424,F 0.095;同seed配对 vs rand_ce Acc +0.066,F -0.078;同seed配对 proto_opsd-本格 Acc -0.024,F +0.003;来源 final.eval.* |
| v3_factorial_heldout_3seed | 随机replay+CE+λ=0.2 OPSD,K=10/段,20% replay,train_holdout holdout_seed=456, seed 42 | InstrDialog++ | 0.439 | 0.442 | 0.070 | 0.068 | 0.560 | 0.552 | 0.593 | 0.550 | 0.000 | 0.000 | 0.442 | 合理:v3h_rand_opsd_instrdialogpp_seed42;Acc 0.439,F 0.070;同seed配对 vs rand_ce Acc +0.045,F -0.057;同seed配对 proto_opsd-本格 Acc -0.013,F +0.014;来源 final.eval.* |
| v3_factorial_heldout_3seed | prototype replay+CE+λ=0.2 OPSD,K=10/段,20% proto replay(10%近邻/10%退化),train_holdout holdout_seed=456, seed 456 | InstrDialog++ | 0.461 | 0.468 | 0.062 | 0.062 | 0.570 | 0.563 | 0.598 | 0.558 | 0.000 | 0.000 | 0.468 | 合理:v3h_proto_opsd_instrdialogpp_seed456;Acc 0.461,F 0.062;同seed配对 vs proto_ce Acc +0.034,F -0.032;同seed配对 vs rand_opsd Acc +0.045,F -0.041;来源 final.eval.* |
| v3_factorial_heldout_3seed | prototype replay+CE+λ=0.2 OPSD,K=10/段,20% proto replay(10%近邻/10%退化),train_holdout holdout_seed=456, seed 789 | InstrDialog++ | 0.400 | 0.403 | 0.097 | 0.097 | 0.512 | 0.505 | 0.568 | 0.503 | 0.100 | 0.100 | 0.403 | 负向:v3h_proto_opsd_instrdialogpp_seed789;Acc 0.400,F 0.097;同seed配对 vs proto_ce Acc -0.032,F +0.030;同seed配对 vs rand_opsd Acc -0.024,F +0.003;来源 final.eval.* |
| v3_factorial_heldout_3seed | prototype replay+CE+λ=0.2 OPSD,K=10/段,20% proto replay(10%近邻/10%退化),train_holdout holdout_seed=456, seed 42 | InstrDialog++ | 0.426 | 0.429 | 0.084 | 0.084 | 0.554 | 0.548 | 0.594 | 0.539 | 0.100 | 0.100 | 0.429 | 合理:v3h_proto_opsd_instrdialogpp_seed42;Acc 0.426,F 0.084;同seed配对 vs proto_ce Acc +0.008,F +0.014;同seed配对 vs rand_opsd Acc -0.013,F +0.014;来源 final.eval.* |

未完成行若存在，续跑整队：

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

单条命令见 `/root/autodl-tmp/rp_lora_v3_runs/logs/<run>.resume_cmd`。不要覆盖已有目录。

## 3-Seed Statistical Summary

样本标准差用 n−1。误差棒和图只在三种子都齐时画。三种子不写显著性，只看方向是否一致。遗忘有利方向是下降。Δ 与 I 都在同 seed、同数据内计算，不拿 789/42 对 456 说方法变好。

### InstrDialog

| Cell | Seen-Avg Acc Mean±SD | Forgetting Mean±SD | n Acc |
|---|---|---|---|
| rand_ce | 0.332±0.033 (n=3) | 0.119±0.026 (n=3) | 3 |
| proto_ce | 0.363±0.037 (n=3) | 0.085±0.031 (n=3) | 3 |
| rand_opsd | 0.368±0.026 (n=3) | 0.080±0.003 (n=3) | 3 |
| proto_opsd | 0.384±0.000 (n=3) | 0.072±0.015 (n=3) | 3 |

| Seed | Δ_PGDR\|CE Acc | Δ_PGDR\|OPSD Acc | Δ_OPSD\|rand Acc | Δ_OPSD\|PGDR Acc | I Acc | Δ_PGDR\|CE F | Δ_OPSD\|rand F | I F |
|---|---|---|---|---|---|---|---|---|
| 456 | +0.042 | -0.011 | +0.074 | +0.021 | -0.053 | -0.044 | -0.050 | +0.056 |
| 789 | +0.095 | +0.016 | +0.063 | -0.016 | -0.079 | -0.083 | -0.061 | +0.067 |
| 42 | -0.042 | +0.042 | -0.026 | +0.058 | +0.084 | +0.028 | -0.006 | -0.044 |

### InstrDialog++

| Cell | Seen-Avg Acc Mean±SD | Forgetting Mean±SD | n Acc |
|---|---|---|---|
| rand_ce | 0.388±0.027 (n=3) | 0.141±0.027 (n=3) | 3 |
| proto_ce | 0.425±0.007 (n=3) | 0.077±0.015 (n=3) | 3 |
| rand_opsd | 0.426±0.012 (n=3) | 0.089±0.017 (n=3) | 3 |
| proto_opsd | 0.429±0.030 (n=3) | 0.081±0.018 (n=3) | 3 |

| Seed | Δ_PGDR\|CE Acc | Δ_PGDR\|OPSD Acc | Δ_OPSD\|rand Acc | Δ_OPSD\|PGDR Acc | I Acc | Δ_PGDR\|CE F | Δ_OPSD\|rand F | I F |
|---|---|---|---|---|---|---|---|---|
| 456 | +0.016 | +0.045 | +0.005 | +0.034 | +0.029 | -0.030 | -0.022 | -0.011 |
| 789 | +0.074 | -0.024 | +0.066 | -0.032 | -0.097 | -0.105 | -0.078 | +0.108 |
| 42 | +0.024 | -0.013 | +0.045 | +0.008 | -0.037 | -0.057 | -0.057 | +0.070 |

## Research Findings and Interpretation

### Q1 PGDR 相对随机回放是否同向？

- InstrDialog Δ_PGDR|CE Acc：+0.042, +0.095, -0.042；方向不一致(2正/1负)。Δ_PGDR|OPSD Acc：-0.011, +0.016, +0.042；方向不一致(2正/1负)。
- InstrDialog++ Δ_PGDR|CE Acc：+0.016, +0.074, +0.024；三种子均为正。Δ_PGDR|OPSD Acc：+0.045, -0.024, -0.013；方向不一致(1正/2负)。
三种子不写显著性。不拿 seed 789/42 对 seed 456 说方法变好。

### Q2 OPSD 相对 CE-only 是否同向？

- InstrDialog Δ_OPSD|random Acc：+0.074, +0.063, -0.026；方向不一致(2正/1负)。Δ_OPSD|PGDR Acc：+0.021, -0.016, +0.058；方向不一致(2正/1负)。
- InstrDialog++ Δ_OPSD|random Acc：+0.005, +0.066, +0.045；三种子均为正。Δ_OPSD|PGDR Acc：+0.034, -0.032, +0.008；方向不一致(2正/1负)。
三种子不写显著性。

### Q3 交互项 I 是否稳定？

- InstrDialog I_Acc：-0.053, -0.079, +0.084；方向不一致(1正/2负)。
- InstrDialog++ I_Acc：+0.029, -0.097, -0.037；方向不一致(1正/2负)。
I = proto_opsd − rand_opsd − proto_ce + rand_ce。缺一格则该 seed 为 —。

### Q4 遗忘是否随 PGDR / OPSD 下降？

遗忘有利方向是下降。
- InstrDialog Δ_PGDR|CE Forgetting：-0.044, -0.083, +0.028；方向不一致(2降/1升)。Δ_OPSD|random Forgetting：-0.050, -0.061, -0.006；三种子均下降。
- InstrDialog++ Δ_PGDR|CE Forgetting：-0.030, -0.105, -0.057；三种子均下降。Δ_OPSD|random Forgetting：-0.022, -0.078, -0.057；三种子均下降。

### Q5 两套 benchmark 故事是否一致？

不一致，只能分流域写。InstrDialog++ 上 Δ_PGDR|CE Acc 三种子为正、Δ_PGDR|CE 与 Δ_OPSD|random 遗忘三种子均下降；InstrDialog 上 Acc 的 PGDR/OPSD 主效应都有 1 个 seed 反号（seed 42），只有 Δ_OPSD|random 遗忘三种子同向下降。不能把一套流的符号套到另一套，也不能写成总效应。

### Q6 Protocol B 相对 Protocol A 只能描述、不能归因泄漏？

Protocol A（`eval_split_monitor`）只作描述性对照。Protocol B 每段 fit 从 50 降到 45，监控 5 条不进梯度；uniform 回放 9、proto 回放 8。A/B 差值混杂了消泄漏与训练条数变化，不能把差值全部写成泄漏效应。近邻槽仍是格式兼容后的均匀抽样，不是余弦检索。
