# v3c 修正配额与真正原型检索

本文件只含 `v3c_matched_budget` 的 36 个 run。一行是一个方法、一个数据集、一颗 seed。
R 是全历史均匀，U 是格式兼容后的均匀 Related 加 train-holdout 退化，P 是显式 L2 余弦 Related 加同一套退化槽。U 不是 cosine prototype。
mixed-batch ρ=0.20 在 45 条 current 上的目标是 11 条回放，实现比例 11/56，不是严格 20%。三种选择器共用这个整数。
旧 `v3h_*` 的 9 对 8 不进入本文件的表、配对差或 Mean±Std。Protocol A 同样分开。
完成 36/36。队列日志里成功 run 的 elapsed 之和是 61388 秒；这个数不是模型指标。

```md
| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | Current-Seg Acc↑ | Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3c_matched_budget | 全历史记忆均匀回放,mixed-batch ρ=0.20,CE(λ=0), seed 456 | InstrDialog | 0.379 | 0.379 | 0.067 | 0.078 | 0.493 | 0.487 | 0.496 | 0.475 | 0.000 | 0.000 | 0.379 | 合理:seed 456同协议;Acc 0.379,F 0.067;run_id v3c_R_ce_instrdialog_seed456;对照 P Acc 0.395,F 0.033(本行−对照 ΔAcc -0.016,ΔF +0.034);对照 U Acc 0.379,F 0.089(本行−对照 ΔAcc +0.000,ΔF -0.022);Δ_OPSD Acc +0.047,F -0.039 |
| v3c_matched_budget | 全历史记忆均匀回放,mixed-batch ρ=0.20,CE(λ=0), seed 789 | InstrDialog | 0.411 | 0.426 | 0.050 | 0.039 | 0.523 | 0.519 | 0.527 | 0.521 | 0.000 | 0.000 | 0.426 | 合理:seed 789同协议;Acc 0.411,F 0.050;run_id v3c_R_ce_instrdialog_seed789;对照 P Acc 0.374,F 0.106(本行−对照 ΔAcc +0.037,ΔF -0.056);对照 U Acc 0.389,F 0.061(本行−对照 ΔAcc +0.022,ΔF -0.011);Δ_OPSD Acc -0.043,F +0.011 |
| v3c_matched_budget | 全历史记忆均匀回放,mixed-batch ρ=0.20,CE(λ=0), seed 42 | InstrDialog | 0.316 | 0.316 | 0.144 | 0.144 | 0.437 | 0.433 | 0.468 | 0.429 | 0.000 | 0.000 | 0.316 | 合理:seed 42同协议;Acc 0.316,F 0.144;run_id v3c_R_ce_instrdialog_seed42;对照 P Acc 0.353,F 0.100(本行−对照 ΔAcc -0.037,ΔF +0.044);对照 U Acc 0.374,F 0.072(本行−对照 ΔAcc -0.058,ΔF +0.072);Δ_OPSD Acc +0.037,F -0.055 |
| v3c_matched_budget | 全历史记忆均匀回放,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 456 | InstrDialog | 0.426 | 0.426 | 0.028 | 0.039 | 0.530 | 0.524 | 0.530 | 0.510 | 0.000 | 0.000 | 0.426 | 合理:seed 456同协议;Acc 0.426,F 0.028;run_id v3c_R_opsd_instrdialog_seed456;对照 P Acc 0.374,F 0.083(本行−对照 ΔAcc +0.052,ΔF -0.055);对照 U Acc 0.395,F 0.061(本行−对照 ΔAcc +0.031,ΔF -0.033);Δ_OPSD Acc +0.047,F -0.039 |
| v3c_matched_budget | 全历史记忆均匀回放,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 789 | InstrDialog | 0.368 | 0.384 | 0.061 | 0.050 | 0.480 | 0.474 | 0.501 | 0.474 | 0.000 | 0.000 | 0.384 | 负向:seed 789同协议;Acc 0.368,F 0.061;run_id v3c_R_opsd_instrdialog_seed789;对照 P Acc 0.395,F 0.056(本行−对照 ΔAcc -0.027,ΔF +0.005);对照 U Acc 0.426,F 0.044(本行−对照 ΔAcc -0.058,ΔF +0.017);Δ_OPSD Acc -0.043,F +0.011 |
| v3c_matched_budget | 全历史记忆均匀回放,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 42 | InstrDialog | 0.353 | 0.358 | 0.089 | 0.094 | 0.466 | 0.461 | 0.496 | 0.462 | 0.000 | 0.000 | 0.358 | 合理:seed 42同协议;Acc 0.353,F 0.089;run_id v3c_R_opsd_instrdialog_seed42;对照 P Acc 0.395,F 0.044(本行−对照 ΔAcc -0.042,ΔF +0.045);对照 U Acc 0.347,F 0.056(本行−对照 ΔAcc +0.006,ΔF +0.033);Δ_OPSD Acc +0.037,F -0.055 |
| v3c_matched_budget | 格式兼容均匀 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE(λ=0), seed 456 | InstrDialog | 0.379 | 0.389 | 0.089 | 0.078 | 0.502 | 0.497 | 0.507 | 0.493 | 0.000 | 0.000 | 0.389 | 负向:seed 456同协议;Acc 0.379,F 0.089;run_id v3c_U_ce_instrdialog_seed456;对照 R Acc 0.379,F 0.067(本行−对照 ΔAcc +0.000,ΔF +0.022);对照 P Acc 0.395,F 0.033(本行−对照 ΔAcc -0.016,ΔF +0.056);Δ_OPSD Acc +0.016,F -0.028 |
| v3c_matched_budget | 格式兼容均匀 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE(λ=0), seed 789 | InstrDialog | 0.389 | 0.405 | 0.061 | 0.050 | 0.494 | 0.491 | 0.514 | 0.490 | 0.000 | 0.000 | 0.405 | 负向:seed 789同协议;Acc 0.389,F 0.061;run_id v3c_U_ce_instrdialog_seed789;对照 R Acc 0.411,F 0.050(本行−对照 ΔAcc -0.022,ΔF +0.011);对照 P Acc 0.374,F 0.106(本行−对照 ΔAcc +0.015,ΔF -0.045);Δ_OPSD Acc +0.037,F -0.017 |
| v3c_matched_budget | 格式兼容均匀 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE(λ=0), seed 42 | InstrDialog | 0.374 | 0.389 | 0.072 | 0.061 | 0.496 | 0.489 | 0.504 | 0.487 | 0.000 | 0.000 | 0.389 | 合理:seed 42同协议;Acc 0.374,F 0.072;run_id v3c_U_ce_instrdialog_seed42;对照 R Acc 0.316,F 0.144(本行−对照 ΔAcc +0.058,ΔF -0.072);对照 P Acc 0.353,F 0.100(本行−对照 ΔAcc +0.021,ΔF -0.028);Δ_OPSD Acc -0.027,F -0.016 |
| v3c_matched_budget | 格式兼容均匀 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 456 | InstrDialog | 0.395 | 0.411 | 0.061 | 0.044 | 0.501 | 0.496 | 0.522 | 0.499 | 0.000 | 0.000 | 0.411 | 负向:seed 456同协议;Acc 0.395,F 0.061;run_id v3c_U_opsd_instrdialog_seed456;对照 R Acc 0.426,F 0.028(本行−对照 ΔAcc -0.031,ΔF +0.033);对照 P Acc 0.374,F 0.083(本行−对照 ΔAcc +0.021,ΔF -0.022);Δ_OPSD Acc +0.016,F -0.028 |
| v3c_matched_budget | 格式兼容均匀 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 789 | InstrDialog | 0.426 | 0.437 | 0.044 | 0.039 | 0.532 | 0.527 | 0.532 | 0.525 | 0.000 | 0.000 | 0.437 | 合理:seed 789同协议;Acc 0.426,F 0.044;run_id v3c_U_opsd_instrdialog_seed789;对照 R Acc 0.368,F 0.061(本行−对照 ΔAcc +0.058,ΔF -0.017);对照 P Acc 0.395,F 0.056(本行−对照 ΔAcc +0.031,ΔF -0.012);Δ_OPSD Acc +0.037,F -0.017 |
| v3c_matched_budget | 格式兼容均匀 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 42 | InstrDialog | 0.347 | 0.358 | 0.056 | 0.050 | 0.451 | 0.445 | 0.491 | 0.441 | 0.000 | 0.000 | 0.358 | 合理:seed 42同协议;Acc 0.347,F 0.056;run_id v3c_U_opsd_instrdialog_seed42;对照 R Acc 0.353,F 0.089(本行−对照 ΔAcc -0.006,ΔF -0.033);对照 P Acc 0.395,F 0.044(本行−对照 ΔAcc -0.048,ΔF +0.012);Δ_OPSD Acc -0.027,F -0.016 |
| v3c_matched_budget | 显式 L2 余弦原型 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE(λ=0), seed 456 | InstrDialog | 0.395 | 0.411 | 0.033 | 0.033 | 0.503 | 0.499 | 0.514 | 0.496 | 0.000 | 0.000 | 0.411 | 合理:seed 456同协议;Acc 0.395,F 0.033;run_id v3c_P_ce_instrdialog_seed456;对照 U Acc 0.379,F 0.089(本行−对照 ΔAcc +0.016,ΔF -0.056);对照 R Acc 0.379,F 0.067(本行−对照 ΔAcc +0.016,ΔF -0.034);Δ_OPSD Acc -0.021,F +0.050 |
| v3c_matched_budget | 显式 L2 余弦原型 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE(λ=0), seed 789 | InstrDialog | 0.374 | 0.379 | 0.106 | 0.100 | 0.474 | 0.469 | 0.493 | 0.472 | 0.000 | 0.000 | 0.379 | 负向:seed 789同协议;Acc 0.374,F 0.106;run_id v3c_P_ce_instrdialog_seed789;对照 U Acc 0.389,F 0.061(本行−对照 ΔAcc -0.015,ΔF +0.045);对照 R Acc 0.411,F 0.050(本行−对照 ΔAcc -0.037,ΔF +0.056);Δ_OPSD Acc +0.021,F -0.050 |
| v3c_matched_budget | 显式 L2 余弦原型 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE(λ=0), seed 42 | InstrDialog | 0.353 | 0.368 | 0.100 | 0.083 | 0.468 | 0.464 | 0.490 | 0.463 | 0.000 | 0.000 | 0.368 | 负向:seed 42同协议;Acc 0.353,F 0.100;run_id v3c_P_ce_instrdialog_seed42;对照 U Acc 0.374,F 0.072(本行−对照 ΔAcc -0.021,ΔF +0.028);对照 R Acc 0.316,F 0.144(本行−对照 ΔAcc +0.037,ΔF -0.044);Δ_OPSD Acc +0.042,F -0.056 |
| v3c_matched_budget | 显式 L2 余弦原型 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 456 | InstrDialog | 0.374 | 0.389 | 0.083 | 0.072 | 0.476 | 0.471 | 0.494 | 0.466 | 0.000 | 0.000 | 0.389 | 负向:seed 456同协议;Acc 0.374,F 0.083;run_id v3c_P_opsd_instrdialog_seed456;对照 U Acc 0.395,F 0.061(本行−对照 ΔAcc -0.021,ΔF +0.022);对照 R Acc 0.426,F 0.028(本行−对照 ΔAcc -0.052,ΔF +0.055);Δ_OPSD Acc -0.021,F +0.050 |
| v3c_matched_budget | 显式 L2 余弦原型 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 789 | InstrDialog | 0.395 | 0.405 | 0.056 | 0.056 | 0.502 | 0.499 | 0.515 | 0.496 | 0.000 | 0.000 | 0.405 | 负向:seed 789同协议;Acc 0.395,F 0.056;run_id v3c_P_opsd_instrdialog_seed789;对照 U Acc 0.426,F 0.044(本行−对照 ΔAcc -0.031,ΔF +0.012);对照 R Acc 0.368,F 0.061(本行−对照 ΔAcc +0.027,ΔF -0.005);Δ_OPSD Acc +0.021,F -0.050 |
| v3c_matched_budget | 显式 L2 余弦原型 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 42 | InstrDialog | 0.395 | 0.411 | 0.044 | 0.039 | 0.517 | 0.512 | 0.531 | 0.517 | 0.000 | 0.000 | 0.411 | 合理:seed 42同协议;Acc 0.395,F 0.044;run_id v3c_P_opsd_instrdialog_seed42;对照 U Acc 0.347,F 0.056(本行−对照 ΔAcc +0.048,ΔF -0.012);对照 R Acc 0.353,F 0.089(本行−对照 ΔAcc +0.042,ΔF -0.045);Δ_OPSD Acc +0.042,F -0.056 |
| v3c_matched_budget | 全历史记忆均匀回放,mixed-batch ρ=0.20,CE(λ=0), seed 456 | InstrDialog++ | 0.434 | 0.439 | 0.100 | 0.095 | 0.561 | 0.554 | 0.600 | 0.554 | 0.100 | 0.100 | 0.439 | 合理:seed 456同协议;Acc 0.434,F 0.100;run_id v3c_R_ce_instrdialogpp_seed456;对照 P Acc 0.400,F 0.103(本行−对照 ΔAcc +0.034,ΔF -0.003);对照 U Acc 0.403,F 0.105(本行−对照 ΔAcc +0.031,ΔF -0.005);Δ_OPSD Acc +0.008,F -0.032 |
| v3c_matched_budget | 全历史记忆均匀回放,mixed-batch ρ=0.20,CE(λ=0), seed 789 | InstrDialog++ | 0.421 | 0.426 | 0.095 | 0.095 | 0.543 | 0.535 | 0.580 | 0.535 | 0.000 | 0.000 | 0.426 | 合理:seed 789同协议;Acc 0.421,F 0.095;run_id v3c_R_ce_instrdialogpp_seed789;对照 P Acc 0.408,F 0.097(本行−对照 ΔAcc +0.013,ΔF -0.002);对照 U Acc 0.418,F 0.097(本行−对照 ΔAcc +0.003,ΔF -0.002);Δ_OPSD Acc +0.008,F -0.009 |
| v3c_matched_budget | 全历史记忆均匀回放,mixed-batch ρ=0.20,CE(λ=0), seed 42 | InstrDialog++ | 0.432 | 0.432 | 0.078 | 0.078 | 0.557 | 0.550 | 0.588 | 0.544 | 0.200 | 0.200 | 0.432 | 合理:seed 42同协议;Acc 0.432,F 0.078;run_id v3c_R_ce_instrdialogpp_seed42;对照 P Acc 0.411,F 0.105(本行−对照 ΔAcc +0.021,ΔF -0.027);对照 U Acc 0.421,F 0.116(本行−对照 ΔAcc +0.011,ΔF -0.038);Δ_OPSD Acc -0.019,F +0.022 |
| v3c_matched_budget | 全历史记忆均匀回放,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 456 | InstrDialog++ | 0.442 | 0.442 | 0.068 | 0.068 | 0.563 | 0.555 | 0.593 | 0.550 | 0.100 | 0.100 | 0.442 | 合理:seed 456同协议;Acc 0.442,F 0.068;run_id v3c_R_opsd_instrdialogpp_seed456;对照 P Acc 0.416,F 0.105(本行−对照 ΔAcc +0.026,ΔF -0.037);对照 U Acc 0.403,F 0.116(本行−对照 ΔAcc +0.039,ΔF -0.048);Δ_OPSD Acc +0.008,F -0.032 |
| v3c_matched_budget | 全历史记忆均匀回放,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 789 | InstrDialog++ | 0.429 | 0.442 | 0.086 | 0.081 | 0.559 | 0.551 | 0.589 | 0.549 | 0.000 | 0.000 | 0.442 | 合理:seed 789同协议;Acc 0.429,F 0.086;run_id v3c_R_opsd_instrdialogpp_seed789;对照 P Acc 0.403,F 0.116(本行−对照 ΔAcc +0.026,ΔF -0.030);对照 U Acc 0.439,F 0.089(本行−对照 ΔAcc -0.010,ΔF -0.003);Δ_OPSD Acc +0.008,F -0.009 |
| v3c_matched_budget | 全历史记忆均匀回放,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 42 | InstrDialog++ | 0.413 | 0.416 | 0.100 | 0.100 | 0.539 | 0.532 | 0.580 | 0.526 | 0.000 | 0.000 | 0.416 | 负向:seed 42同协议;Acc 0.413,F 0.100;run_id v3c_R_opsd_instrdialogpp_seed42;对照 P Acc 0.405,F 0.105(本行−对照 ΔAcc +0.008,ΔF -0.005);对照 U Acc 0.405,F 0.116(本行−对照 ΔAcc +0.008,ΔF -0.016);Δ_OPSD Acc -0.019,F +0.022 |
| v3c_matched_budget | 格式兼容均匀 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE(λ=0), seed 456 | InstrDialog++ | 0.403 | 0.405 | 0.105 | 0.105 | 0.524 | 0.517 | 0.574 | 0.514 | 0.000 | 0.000 | 0.405 | 负向:seed 456同协议;Acc 0.403,F 0.105;run_id v3c_U_ce_instrdialogpp_seed456;对照 R Acc 0.434,F 0.100(本行−对照 ΔAcc -0.031,ΔF +0.005);对照 P Acc 0.400,F 0.103(本行−对照 ΔAcc +0.003,ΔF +0.002);Δ_OPSD Acc +0.000,F +0.011 |
| v3c_matched_budget | 格式兼容均匀 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE(λ=0), seed 789 | InstrDialog++ | 0.418 | 0.421 | 0.097 | 0.095 | 0.537 | 0.530 | 0.580 | 0.526 | 0.000 | 0.000 | 0.421 | 合理:seed 789同协议;Acc 0.418,F 0.097;run_id v3c_U_ce_instrdialogpp_seed789;对照 R Acc 0.421,F 0.095(本行−对照 ΔAcc -0.003,ΔF +0.002);对照 P Acc 0.408,F 0.097(本行−对照 ΔAcc +0.010,ΔF +0.000);Δ_OPSD Acc +0.021,F -0.008 |
| v3c_matched_budget | 格式兼容均匀 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE(λ=0), seed 42 | InstrDialog++ | 0.421 | 0.421 | 0.116 | 0.116 | 0.535 | 0.529 | 0.578 | 0.528 | 0.100 | 0.100 | 0.421 | 负向:seed 42同协议;Acc 0.421,F 0.116;run_id v3c_U_ce_instrdialogpp_seed42;对照 R Acc 0.432,F 0.078(本行−对照 ΔAcc -0.011,ΔF +0.038);对照 P Acc 0.411,F 0.105(本行−对照 ΔAcc +0.010,ΔF +0.011);Δ_OPSD Acc -0.016,F +0.000 |
| v3c_matched_budget | 格式兼容均匀 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 456 | InstrDialog++ | 0.403 | 0.408 | 0.116 | 0.114 | 0.520 | 0.512 | 0.570 | 0.509 | 0.000 | 0.000 | 0.408 | 负向:seed 456同协议;Acc 0.403,F 0.116;run_id v3c_U_opsd_instrdialogpp_seed456;对照 R Acc 0.442,F 0.068(本行−对照 ΔAcc -0.039,ΔF +0.048);对照 P Acc 0.416,F 0.105(本行−对照 ΔAcc -0.013,ΔF +0.011);Δ_OPSD Acc +0.000,F +0.011 |
| v3c_matched_budget | 格式兼容均匀 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 789 | InstrDialog++ | 0.439 | 0.445 | 0.089 | 0.086 | 0.556 | 0.549 | 0.589 | 0.549 | 0.000 | 0.000 | 0.445 | 合理:seed 789同协议;Acc 0.439,F 0.089;run_id v3c_U_opsd_instrdialogpp_seed789;对照 R Acc 0.429,F 0.086(本行−对照 ΔAcc +0.010,ΔF +0.003);对照 P Acc 0.403,F 0.116(本行−对照 ΔAcc +0.036,ΔF -0.027);Δ_OPSD Acc +0.021,F -0.008 |
| v3c_matched_budget | 格式兼容均匀 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 42 | InstrDialog++ | 0.405 | 0.411 | 0.116 | 0.114 | 0.529 | 0.523 | 0.580 | 0.526 | 0.100 | 0.100 | 0.411 | 负向:seed 42同协议;Acc 0.405,F 0.116;run_id v3c_U_opsd_instrdialogpp_seed42;对照 R Acc 0.413,F 0.100(本行−对照 ΔAcc -0.008,ΔF +0.016);对照 P Acc 0.405,F 0.105(本行−对照 ΔAcc +0.000,ΔF +0.011);Δ_OPSD Acc -0.016,F +0.000 |
| v3c_matched_budget | 显式 L2 余弦原型 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE(λ=0), seed 456 | InstrDialog++ | 0.400 | 0.403 | 0.103 | 0.100 | 0.525 | 0.517 | 0.571 | 0.513 | 0.100 | 0.100 | 0.403 | 负向:seed 456同协议;Acc 0.400,F 0.103;run_id v3c_P_ce_instrdialogpp_seed456;对照 U Acc 0.403,F 0.105(本行−对照 ΔAcc -0.003,ΔF -0.002);对照 R Acc 0.434,F 0.100(本行−对照 ΔAcc -0.034,ΔF +0.003);Δ_OPSD Acc +0.016,F +0.002 |
| v3c_matched_budget | 显式 L2 余弦原型 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE(λ=0), seed 789 | InstrDialog++ | 0.408 | 0.408 | 0.097 | 0.097 | 0.524 | 0.515 | 0.579 | 0.512 | 0.100 | 0.100 | 0.408 | 负向:seed 789同协议;Acc 0.408,F 0.097;run_id v3c_P_ce_instrdialogpp_seed789;对照 U Acc 0.418,F 0.097(本行−对照 ΔAcc -0.010,ΔF +0.000);对照 R Acc 0.421,F 0.095(本行−对照 ΔAcc -0.013,ΔF +0.002);Δ_OPSD Acc -0.005,F +0.019 |
| v3c_matched_budget | 显式 L2 余弦原型 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE(λ=0), seed 42 | InstrDialog++ | 0.411 | 0.413 | 0.105 | 0.103 | 0.536 | 0.530 | 0.580 | 0.525 | 0.100 | 0.100 | 0.413 | 负向:seed 42同协议;Acc 0.411,F 0.105;run_id v3c_P_ce_instrdialogpp_seed42;对照 U Acc 0.421,F 0.116(本行−对照 ΔAcc -0.010,ΔF -0.011);对照 R Acc 0.432,F 0.078(本行−对照 ΔAcc -0.021,ΔF +0.027);Δ_OPSD Acc -0.006,F +0.000 |
| v3c_matched_budget | 显式 L2 余弦原型 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 456 | InstrDialog++ | 0.416 | 0.416 | 0.105 | 0.108 | 0.538 | 0.532 | 0.570 | 0.528 | 0.100 | 0.100 | 0.416 | 负向:seed 456同协议;Acc 0.416,F 0.105;run_id v3c_P_opsd_instrdialogpp_seed456;对照 U Acc 0.403,F 0.116(本行−对照 ΔAcc +0.013,ΔF -0.011);对照 R Acc 0.442,F 0.068(本行−对照 ΔAcc -0.026,ΔF +0.037);Δ_OPSD Acc +0.016,F +0.002 |
| v3c_matched_budget | 显式 L2 余弦原型 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 789 | InstrDialog++ | 0.403 | 0.405 | 0.116 | 0.116 | 0.519 | 0.512 | 0.576 | 0.509 | 0.100 | 0.100 | 0.405 | 负向:seed 789同协议;Acc 0.403,F 0.116;run_id v3c_P_opsd_instrdialogpp_seed789;对照 U Acc 0.439,F 0.089(本行−对照 ΔAcc -0.036,ΔF +0.027);对照 R Acc 0.429,F 0.086(本行−对照 ΔAcc -0.026,ΔF +0.030);Δ_OPSD Acc -0.005,F +0.019 |
| v3c_matched_budget | 显式 L2 余弦原型 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20,CE+λ=0.2 OPSD, seed 42 | InstrDialog++ | 0.405 | 0.405 | 0.105 | 0.108 | 0.525 | 0.518 | 0.570 | 0.516 | 0.000 | 0.000 | 0.405 | 合理:seed 42同协议;Acc 0.405,F 0.105;run_id v3c_P_opsd_instrdialogpp_seed42;对照 U Acc 0.405,F 0.116(本行−对照 ΔAcc +0.000,ΔF -0.011);对照 R Acc 0.413,F 0.100(本行−对照 ΔAcc -0.008,ΔF +0.005);Δ_OPSD Acc -0.006,F +0.000 |
```

## 配对效应

只在 v3c 同 seed、同数据、同目标的格子上做差。差由上表三位小数相减。
遗忘下降才是改善：ΔF 为负才算遗忘变好。P 低于 U 或 R 的差保留负号。
Δ_prototype = P−U，Δ_PGDR = P−R，Δ_OPSD = OPSD−CE。
交互项 I = (P 的 Δ_OPSD) − (R 的 Δ_OPSD)。I 只取决于数据和 seed，同一 seed 的 CE 行与 CE+OPSD 行写同一个 I。
Protocol A 和 v3h 的 9 对 8 不进入这些差，也不进入后面的 Mean±Std。

| Data | Objective | Seed | Δ_prototype Acc | Δ_prototype F | Δ_PGDR Acc | Δ_PGDR F | Δ_OPSD(P) Acc | Δ_OPSD(P) F | I Acc | I F |
|---|---|---|---|---|---|---|---|---|---|---|
| InstrDialog | CE | 456 | +0.016 | -0.056 | +0.016 | -0.034 | -0.021 | +0.050 | -0.068 | +0.089 |
| InstrDialog | CE | 789 | -0.015 | +0.045 | -0.037 | +0.056 | +0.021 | -0.050 | +0.064 | -0.061 |
| InstrDialog | CE | 42 | -0.021 | +0.028 | +0.037 | -0.044 | +0.042 | -0.056 | +0.005 | -0.001 |
| InstrDialog | CE+OPSD | 456 | -0.021 | +0.022 | -0.052 | +0.055 | -0.021 | +0.050 | -0.068 | +0.089 |
| InstrDialog | CE+OPSD | 789 | -0.031 | +0.012 | +0.027 | -0.005 | +0.021 | -0.050 | +0.064 | -0.061 |
| InstrDialog | CE+OPSD | 42 | +0.048 | -0.012 | +0.042 | -0.045 | +0.042 | -0.056 | +0.005 | -0.001 |
| InstrDialog++ | CE | 456 | -0.003 | -0.002 | -0.034 | +0.003 | +0.016 | +0.002 | +0.008 | +0.034 |
| InstrDialog++ | CE | 789 | -0.010 | +0.000 | -0.013 | +0.002 | -0.005 | +0.019 | -0.013 | +0.028 |
| InstrDialog++ | CE | 42 | -0.010 | -0.011 | -0.021 | +0.027 | -0.006 | +0.000 | +0.013 | -0.022 |
| InstrDialog++ | CE+OPSD | 456 | +0.013 | -0.011 | -0.026 | +0.037 | +0.016 | +0.002 | +0.008 | +0.034 |
| InstrDialog++ | CE+OPSD | 789 | -0.036 | +0.027 | -0.026 | +0.030 | -0.005 | +0.019 | -0.013 | +0.028 |
| InstrDialog++ | CE+OPSD | 42 | +0.000 | -0.011 | -0.008 | +0.005 | -0.006 | +0.000 | +0.013 | -0.022 |

## 三种子 Mean±样本标准差

每一格都要有 seed 456、789、42 的 `final.eval` 才计算。样本标准差分母是 n−1。
均值和标准差用未四舍五入的原值计算，再保留三位。只含 v3c_matched_budget。

| Data | Letter | Objective | n | Acc mean | Acc std | F mean | F std |
|---|---|---|---|---|---|---|---|
| InstrDialog | R | CE | 3 | 0.368 | 0.048 | 0.087 | 0.050 |
| InstrDialog | R | CE+OPSD | 3 | 0.382 | 0.039 | 0.059 | 0.031 |
| InstrDialog | U | CE | 3 | 0.381 | 0.008 | 0.074 | 0.014 |
| InstrDialog | U | CE+OPSD | 3 | 0.389 | 0.040 | 0.054 | 0.008 |
| InstrDialog | P | CE | 3 | 0.374 | 0.021 | 0.080 | 0.040 |
| InstrDialog | P | CE+OPSD | 3 | 0.388 | 0.012 | 0.061 | 0.020 |
| InstrDialog++ | R | CE | 3 | 0.429 | 0.007 | 0.091 | 0.011 |
| InstrDialog++ | R | CE+OPSD | 3 | 0.428 | 0.014 | 0.085 | 0.016 |
| InstrDialog++ | U | CE | 3 | 0.414 | 0.010 | 0.106 | 0.009 |
| InstrDialog++ | U | CE+OPSD | 3 | 0.416 | 0.021 | 0.107 | 0.016 |
| InstrDialog++ | P | CE | 3 | 0.406 | 0.005 | 0.102 | 0.004 |
| InstrDialog++ | P | CE+OPSD | 3 | 0.408 | 0.007 | 0.109 | 0.006 |

## 效应的三种子均值

这里的差先按上表三位小数相减，再对三颗 seed 取均值和样本标准差。
ΔF 为负才是遗忘改善。

| Data | Objective | Δ_prototype Acc | Δ_prototype F | Δ_PGDR Acc | Δ_PGDR F |
|---|---|---|---|---|---|
| InstrDialog | CE | -0.007±0.020 | 0.006±0.054 | 0.005±0.038 | -0.007±0.055 |
| InstrDialog | CE+OPSD | -0.001±0.043 | 0.007±0.017 | 0.006±0.051 | 0.002±0.050 |
| InstrDialog++ | CE | -0.008±0.004 | -0.004±0.006 | -0.023±0.011 | 0.011±0.014 |
| InstrDialog++ | CE+OPSD | -0.008±0.025 | 0.002±0.022 | -0.020±0.010 | 0.024±0.017 |

| Data | Letter | Δ_OPSD Acc | Δ_OPSD F |
|---|---|---|---|
| InstrDialog | R | 0.014±0.049 | -0.028±0.034 |
| InstrDialog | U | 0.009±0.033 | -0.020±0.007 |
| InstrDialog | P | 0.014±0.032 | -0.019±0.060 |
| InstrDialog++ | R | -0.001±0.016 | -0.006±0.027 |
| InstrDialog++ | U | 0.002±0.019 | 0.001±0.010 |
| InstrDialog++ | P | 0.002±0.012 | 0.007±0.010 |

| Data | I Acc | I F |
|---|---|---|
| InstrDialog | 0.000±0.066 | 0.009±0.075 |
| InstrDialog++ | 0.003±0.014 | 0.013±0.031 |

## Q1–Q2

Q1 问配额对齐之后，真余弦 PGDR（P）相对全历史随机（R）还有没有优势。对照量是 Δ_PGDR = P−R。
Q2 问真余弦（P）相对 compat_uniform（U）还有没有优势。U 是格式兼容之后均匀抽 Related，再加上和 P 相同的 train-holdout 退化槽。U 不是 cosine prototype。对照量是 Δ_prototype = P−U。
两问都只使用上面 36 个 v3c run。遗忘下降才算改善。P 不如 U、也不如 R 的格子按负数写下，不取绝对值，也不改成「持平」。

### Q1 配额公平后 PGDR 相对随机

合理:InstrDialog CE 上 Δ_PGDR 的 Acc 为 +0.005±0.038（seed 456 +0.016、seed 789 -0.037、seed 42 +0.037），F 为 -0.007±0.055（seed 456 -0.034、seed 789 +0.056、seed 42 -0.044）。三种子符号不完全一致，或均值接近 0。不能写成稳定优于对照。

合理:InstrDialog CE+OPSD 上 Δ_PGDR 的 Acc 为 +0.006±0.051（seed 456 -0.052、seed 789 +0.027、seed 42 +0.042），F 为 +0.002±0.050（seed 456 +0.055、seed 789 -0.005、seed 42 -0.045）。三种子符号不完全一致，或均值接近 0。不能写成稳定优于对照。

负向:InstrDialog++ CE 上 Δ_PGDR 的 Acc 为 -0.023±0.011（seed 456 -0.034、seed 789 -0.013、seed 42 -0.021），F 为 +0.011±0.014（seed 456 +0.003、seed 789 +0.002、seed 42 +0.027）。三颗 seed 都是 P 的准确率更低，而且遗忘都更高。遗忘没有下降。

负向:InstrDialog++ CE+OPSD 上 Δ_PGDR 的 Acc 为 -0.020±0.010（seed 456 -0.026、seed 789 -0.026、seed 42 -0.008），F 为 +0.024±0.017（seed 456 +0.037、seed 789 +0.030、seed 42 +0.005）。三颗 seed 都是 P 的准确率更低，而且遗忘都更高。遗忘没有下降。

配额公平之后，P 没有稳定超过 R。InstrDialog 上三颗 seed 的符号相反，均值跨过 0。InstrDialog++ 的 CE 和 CE+OPSD 都是三颗 seed 的 P 准确率更低，而且遗忘都更高。这是负向结果，不是检索分数可以改写的结果。

### Q2 真余弦相对 compat_uniform

合理:InstrDialog CE 上 Δ_prototype 的 Acc 为 -0.007±0.020（seed 456 +0.016、seed 789 -0.015、seed 42 -0.021），F 为 +0.006±0.054（seed 456 -0.056、seed 789 +0.045、seed 42 +0.028）。三种子符号不完全一致，或均值接近 0。不能写成稳定优于对照。

合理:InstrDialog CE+OPSD 上 Δ_prototype 的 Acc 为 -0.001±0.043（seed 456 -0.021、seed 789 -0.031、seed 42 +0.048），F 为 +0.007±0.017（seed 456 +0.022、seed 789 +0.012、seed 42 -0.012）。三种子符号不完全一致，或均值接近 0。不能写成稳定优于对照。

负向:InstrDialog++ CE 上 Δ_prototype 的 Acc 为 -0.008±0.004（seed 456 -0.003、seed 789 -0.010、seed 42 -0.010），F 为 -0.004±0.006（seed 456 -0.002、seed 789 +0.000、seed 42 -0.011）。三颗 seed 的准确率都是 P 更低。遗忘不是稳定下降。

合理:InstrDialog++ CE+OPSD 上 Δ_prototype 的 Acc 为 -0.008±0.025（seed 456 +0.013、seed 789 -0.036、seed 42 +0.000），F 为 +0.002±0.022（seed 456 -0.011、seed 789 +0.027、seed 42 -0.011）。三种子符号不完全一致，或均值接近 0。不能写成稳定优于对照。

真余弦没有稳定超过 compat_uniform。InstrDialog CE 只有 seed 456 是 P 更好，789 和 42 是 P 更差。InstrDialog CE+OPSD 只有 seed 42 的准确率是 P 更高，另外两颗是 P 更低，遗忘均值也没有下降。InstrDialog++ CE 三颗 seed 的准确率都是 P 低于 U，遗忘只是略降。InstrDialog++ CE+OPSD 的 seed 789 上 P 明显更差。这些差原样留在表里。

### Δ_OPSD 与交互项 I

Δ_OPSD = 同一选择器的 CE+OPSD 减去 CE。I = P 的 Δ_OPSD 减去 R 的 Δ_OPSD。I 的准确率为正，表示 OPSD 给 P 的准确率提升大于给 R 的提升。I 的遗忘为正，表示 OPSD 使 P 的遗忘比 R 增加得更多，这不是改善。

InstrDialog R 的 Δ_OPSD Acc +0.014±0.049（seed 456 +0.047、seed 789 -0.043、seed 42 +0.037），F -0.028±0.034（seed 456 -0.039、seed 789 +0.011、seed 42 -0.055）。
InstrDialog U 的 Δ_OPSD Acc +0.009±0.033（seed 456 +0.016、seed 789 +0.037、seed 42 -0.027），F -0.020±0.007（seed 456 -0.028、seed 789 -0.017、seed 42 -0.016）。
InstrDialog P 的 Δ_OPSD Acc +0.014±0.032（seed 456 -0.021、seed 789 +0.021、seed 42 +0.042），F -0.019±0.060（seed 456 +0.050、seed 789 -0.050、seed 42 -0.056）。
InstrDialog++ R 的 Δ_OPSD Acc -0.001±0.016（seed 456 +0.008、seed 789 +0.008、seed 42 -0.019），F -0.006±0.027（seed 456 -0.032、seed 789 -0.009、seed 42 +0.022）。
InstrDialog++ U 的 Δ_OPSD Acc +0.002±0.019（seed 456 +0.000、seed 789 +0.021、seed 42 -0.016），F +0.001±0.010（seed 456 +0.011、seed 789 -0.008、seed 42 +0.000）。
InstrDialog++ P 的 Δ_OPSD Acc +0.002±0.012（seed 456 +0.016、seed 789 -0.005、seed 42 -0.006），F +0.007±0.010（seed 456 +0.002、seed 789 +0.019、seed 42 +0.000）。

InstrDialog 的 I Acc +0.000±0.066（seed 456 -0.068、seed 789 +0.064、seed 42 +0.005），I F +0.009±0.075（seed 456 +0.089、seed 789 -0.061、seed 42 -0.001）。
InstrDialog++ 的 I Acc +0.003±0.014（seed 456 +0.008、seed 789 -0.013、seed 42 +0.013），I F +0.013±0.031（seed 456 +0.034、seed 789 +0.028、seed 42 -0.022）。

InstrDialog 上 I 的准确率均值接近 0，seed 456 为负、seed 789 为正。InstrDialog++ 上 I 的遗忘均值为正，OPSD 没有把 P 的遗忘压到比 R 更低的增量上。不能把 I 写成 PGDR 与 OPSD 的正交互。

## 回放日志里的配额和余弦

下面只统计 v3c `replay_log.jsonl`。segment 0 没有历史记忆，不拿来核对目标条数。

R 在 segment_id>0 的 n_replay_target 取值 [11]；selected_cosine_scores 共 3618 个，均值 0.761，样本标准差 0.079。
U 在 segment_id>0 的 n_replay_target 取值 [11]；selected_cosine_scores 共 3618 个，均值 0.767，样本标准差 0.075。
P 在 segment_id>0 的 n_replay_target 取值 [11]；selected_cosine_scores 共 3618 个，均值 0.797，样本标准差 0.069。

P 的选中余弦均值高于 U 和 R，只说明被抽中的段在冻结指令向量上更接近当前段。它不是 Seen-Avg Acc，也不是遗忘下降。R 和 U 的余弦是日志里顺手记下的，选择时不用这个分数。

## 图

图由 `results/figures/plot_v3c_corrected.py` 只根据已有 `final_metrics.json` 或 `replay_log.jsonl` 绘制。没有实测点的图不写文件。

- `results/figures/prototype_vs_uniform.pdf`：同 seed 的 P−U。
- `results/figures/budget_matched_comparison.pdf`：R/U/P 在两种目标上的实测均值，柱上标 n。
- `results/figures/factorial_accuracy.pdf`：每颗 seed 的 CE 到 CE+OPSD 准确率。
- `results/figures/factorial_forgetting.pdf`：同一因子的遗忘。
- `results/figures/selected_cosine_distribution.pdf`：日志里的 selected L2 cosine。没有该字段就不画。
- `results/figures/per_seed_effects.pdf`：每颗 seed 的 P−R 与 P−U。

