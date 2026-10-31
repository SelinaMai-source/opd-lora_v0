# ABLATION v3 实验结果与待补表

数字只来自 `final_metrics.json`。缺测写 —。旧协议 `v3_` 使用官方 eval 遗忘选样（`eval_split_monitor`）。`v3h_` 使用训练集留出 5/50（`train_holdout`），四个方法共用同一划分。两套协议分行，不与旧 0.474/0.450 混成同一协议。

遗忘列保持原始定义（越小越好）。交互表最后一列只对遗忘类指标给出符号翻转后的“越大越好”值，图中遗忘也取该翻转。梯度余弦未在 8B 上测量（保留计算图会顶满显存），不写数值。

已填实测 run 数：32 / 32。

## 11 指标：旧协议 seed 456（已有，不重跑）

| run | monitor | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3_rand_ce_instrdialog_seed456 | eval_split_monitor | 0.395 | 0.395 | 0.072 | 0.078 | 0.504 | 0.500 | 0.521 | 0.495 | 0.000 | 0.000 | 0.395 |
| v3_proto_ce_instrdialog_seed456 | eval_split_monitor | 0.384 | 0.395 | 0.050 | 0.056 | 0.494 | 0.488 | 0.506 | 0.492 | 0.000 | 0.000 | 0.395 |
| v3_rand_opsd_instrdialog_seed456 | eval_split_monitor | 0.400 | 0.411 | 0.044 | 0.044 | 0.502 | 0.498 | 0.515 | 0.491 | 0.000 | 0.000 | 0.411 |
| v3_proto_opsd_instrdialog_seed456 | eval_split_monitor | 0.432 | 0.447 | 0.017 | 0.011 | 0.531 | 0.527 | 0.543 | 0.522 | 0.000 | 0.000 | 0.447 |
| v3_rand_ce_instrdialogpp_seed456 | eval_split_monitor | 0.376 | 0.376 | 0.130 | 0.130 | 0.495 | 0.485 | 0.542 | 0.481 | 0.000 | 0.000 | 0.376 |
| v3_proto_ce_instrdialogpp_seed456 | eval_split_monitor | 0.474 | 0.474 | 0.035 | 0.038 | 0.594 | 0.587 | 0.618 | 0.582 | 0.000 | 0.000 | 0.474 |
| v3_rand_opsd_instrdialogpp_seed456 | eval_split_monitor | 0.389 | 0.395 | 0.116 | 0.114 | 0.505 | 0.496 | 0.549 | 0.498 | 0.100 | 0.100 | 0.395 |
| v3_proto_opsd_instrdialogpp_seed456 | eval_split_monitor | 0.450 | 0.453 | 0.068 | 0.068 | 0.572 | 0.564 | 0.601 | 0.563 | 0.100 | 0.100 | 0.453 |

## 11 指标：旧协议 seed 789 / 42

| run | monitor | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3_rand_ce_instrdialog_seed789 | eval_split_monitor | 0.368 | 0.379 | 0.078 | 0.067 | 0.493 | 0.489 | 0.504 | 0.493 | 0.000 | 0.000 | 0.379 |
| v3_rand_ce_instrdialog_seed42 | eval_split_monitor | 0.326 | 0.332 | 0.122 | 0.122 | 0.444 | 0.436 | 0.465 | 0.434 | 0.000 | 0.000 | 0.332 |
| v3_proto_ce_instrdialog_seed789 | eval_split_monitor | 0.374 | 0.384 | 0.056 | 0.050 | 0.480 | 0.475 | 0.501 | 0.477 | 0.000 | 0.000 | 0.384 |
| v3_proto_ce_instrdialog_seed42 | eval_split_monitor | 0.411 | 0.426 | 0.039 | 0.022 | 0.525 | 0.519 | 0.526 | 0.521 | 0.000 | 0.000 | 0.426 |
| v3_rand_opsd_instrdialog_seed789 | eval_split_monitor | 0.368 | 0.379 | 0.083 | 0.072 | 0.480 | 0.476 | 0.491 | 0.480 | 0.000 | 0.000 | 0.379 |
| v3_rand_opsd_instrdialog_seed42 | eval_split_monitor | 0.321 | 0.332 | 0.122 | 0.117 | 0.436 | 0.429 | 0.465 | 0.425 | 0.000 | 0.000 | 0.332 |
| v3_proto_opsd_instrdialog_seed789 | eval_split_monitor | 0.384 | 0.400 | 0.050 | 0.039 | 0.496 | 0.491 | 0.515 | 0.491 | 0.000 | 0.000 | 0.400 |
| v3_proto_opsd_instrdialog_seed42 | eval_split_monitor | 0.368 | 0.379 | 0.067 | 0.056 | 0.476 | 0.469 | 0.498 | 0.465 | 0.000 | 0.000 | 0.379 |
| v3_rand_ce_instrdialogpp_seed789 | eval_split_monitor | 0.389 | 0.392 | 0.127 | 0.127 | 0.515 | 0.508 | 0.563 | 0.499 | 0.000 | 0.000 | 0.392 |
| v3_rand_ce_instrdialogpp_seed42 | eval_split_monitor | 0.416 | 0.418 | 0.089 | 0.089 | 0.526 | 0.519 | 0.576 | 0.519 | 0.300 | 0.300 | 0.418 |
| v3_proto_ce_instrdialogpp_seed789 | eval_split_monitor | 0.434 | 0.434 | 0.089 | 0.089 | 0.551 | 0.544 | 0.585 | 0.539 | 0.000 | 0.000 | 0.434 |
| v3_proto_ce_instrdialogpp_seed42 | eval_split_monitor | 0.453 | 0.461 | 0.076 | 0.068 | 0.569 | 0.561 | 0.603 | 0.562 | 0.100 | 0.100 | 0.461 |
| v3_rand_opsd_instrdialogpp_seed789 | eval_split_monitor | 0.411 | 0.413 | 0.100 | 0.100 | 0.552 | 0.543 | 0.586 | 0.539 | 0.100 | 0.100 | 0.413 |
| v3_rand_opsd_instrdialogpp_seed42 | eval_split_monitor | 0.400 | 0.403 | 0.105 | 0.103 | 0.508 | 0.500 | 0.557 | 0.498 | 0.100 | 0.100 | 0.403 |
| v3_proto_opsd_instrdialogpp_seed789 | eval_split_monitor | 0.455 | 0.455 | 0.049 | 0.049 | 0.555 | 0.550 | 0.597 | 0.546 | 0.100 | 0.100 | 0.455 |
| v3_proto_opsd_instrdialogpp_seed42 | eval_split_monitor | 0.442 | 0.445 | 0.073 | 0.070 | 0.564 | 0.556 | 0.592 | 0.559 | 0.000 | 0.000 | 0.445 |

## 11 指标：v3h 训练集留出监控 seed 456

| run | monitor | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3h_rand_ce_instrdialog_seed456 | train_holdout | 0.321 | 0.326 | 0.128 | 0.133 | 0.408 | 0.404 | 0.450 | 0.398 | 0.000 | 0.000 | 0.326 |
| v3h_proto_ce_instrdialog_seed456 | train_holdout | 0.363 | 0.379 | 0.083 | 0.072 | 0.455 | 0.451 | 0.490 | 0.442 | 0.000 | 0.000 | 0.379 |
| v3h_rand_opsd_instrdialog_seed456 | train_holdout | 0.395 | 0.395 | 0.078 | 0.078 | 0.498 | 0.493 | 0.513 | 0.491 | 0.000 | 0.000 | 0.395 |
| v3h_proto_opsd_instrdialog_seed456 | train_holdout | 0.384 | 0.389 | 0.089 | 0.083 | 0.466 | 0.460 | 0.484 | 0.452 | 0.000 | 0.000 | 0.389 |
| v3h_rand_ce_instrdialogpp_seed456 | train_holdout | 0.411 | 0.411 | 0.124 | 0.127 | 0.543 | 0.535 | 0.580 | 0.534 | 0.200 | 0.200 | 0.411 |
| v3h_proto_ce_instrdialogpp_seed456 | train_holdout | 0.426 | 0.429 | 0.095 | 0.095 | 0.537 | 0.529 | 0.569 | 0.525 | 0.000 | 0.000 | 0.429 |
| v3h_rand_opsd_instrdialogpp_seed456 | train_holdout | 0.416 | 0.421 | 0.103 | 0.103 | 0.540 | 0.532 | 0.581 | 0.529 | 0.200 | 0.200 | 0.421 |
| v3h_proto_opsd_instrdialogpp_seed456 | train_holdout | 0.461 | 0.468 | 0.062 | 0.062 | 0.570 | 0.563 | 0.598 | 0.558 | 0.000 | 0.000 | 0.468 |

## E1 交互

交互 = proto_opsd − rand_opsd − proto_ce + rand_ce。proto 主效应与 OPSD 主效应是两条边的平均。缺任一格则整行交互为 —。

### seed 456 旧协议

| Data | metric | rand_ce | proto_ce | rand_opsd | proto_opsd | interaction | proto main | OPSD main | forgetting sign-flip |
|---|---|---|---|---|---|---|---|---|---|
| instrdialog | Seen-Avg Acc | 0.395 | 0.384 | 0.400 | 0.432 | 0.042 | 0.011 | 0.026 |  |
| instrdialog | Seen-Avg Task-aware Acc | 0.395 | 0.395 | 0.411 | 0.447 | 0.037 | 0.018 | 0.034 |  |
| instrdialog | Forgetting | 0.072 | 0.050 | 0.044 | 0.017 | -0.006 | -0.025 | -0.031 | 0.006 |
| instrdialog | Task-Aware Forgetting | 0.078 | 0.056 | 0.044 | 0.011 | -0.011 | -0.028 | -0.039 | 0.011 |
| instrdialog | Token F1 | 0.504 | 0.494 | 0.502 | 0.531 | 0.039 | 0.009 | 0.017 |  |
| instrdialog | ROUGE-L | 0.500 | 0.488 | 0.498 | 0.527 | 0.041 | 0.009 | 0.018 |  |
| instrdialog | BLEU | 0.521 | 0.506 | 0.515 | 0.543 | 0.043 | 0.007 | 0.015 |  |
| instrdialog | LCS Overlap | 0.495 | 0.492 | 0.491 | 0.522 | 0.034 | 0.014 | 0.013 |  |
| instrdialog | Current-Seg Acc | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| instrdialog | Current-Seg Task-aware Acc | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| instrdialog | Task-aware Score Mean | 0.395 | 0.395 | 0.411 | 0.447 | 0.037 | 0.018 | 0.034 |  |
| instrdialogpp | Seen-Avg Acc | 0.376 | 0.474 | 0.389 | 0.450 | -0.037 | 0.079 | -0.005 |  |
| instrdialogpp | Seen-Avg Task-aware Acc | 0.376 | 0.474 | 0.395 | 0.453 | -0.039 | 0.078 | -0.001 |  |
| instrdialogpp | Forgetting | 0.130 | 0.035 | 0.116 | 0.068 | 0.046 | -0.072 | 0.009 | -0.046 |
| instrdialogpp | Task-Aware Forgetting | 0.130 | 0.038 | 0.114 | 0.068 | 0.046 | -0.069 | 0.007 | -0.046 |
| instrdialogpp | Token F1 | 0.495 | 0.594 | 0.505 | 0.572 | -0.032 | 0.083 | -0.006 |  |
| instrdialogpp | ROUGE-L | 0.485 | 0.587 | 0.496 | 0.564 | -0.034 | 0.085 | -0.006 |  |
| instrdialogpp | BLEU | 0.542 | 0.618 | 0.549 | 0.601 | -0.024 | 0.064 | -0.005 |  |
| instrdialogpp | LCS Overlap | 0.481 | 0.582 | 0.498 | 0.563 | -0.036 | 0.083 | -0.001 |  |
| instrdialogpp | Current-Seg Acc | 0.000 | 0.000 | 0.100 | 0.100 | 0.000 | 0.000 | 0.100 |  |
| instrdialogpp | Current-Seg Task-aware Acc | 0.000 | 0.000 | 0.100 | 0.100 | 0.000 | 0.000 | 0.100 |  |
| instrdialogpp | Task-aware Score Mean | 0.376 | 0.474 | 0.395 | 0.453 | -0.039 | 0.078 | -0.001 |  |

### seed 456 v3h（训练集留出监控）

| Data | metric | rand_ce | proto_ce | rand_opsd | proto_opsd | interaction | proto main | OPSD main | forgetting sign-flip |
|---|---|---|---|---|---|---|---|---|---|
| instrdialog | Seen-Avg Acc | 0.321 | 0.363 | 0.395 | 0.384 | -0.053 | 0.016 | 0.047 |  |
| instrdialog | Seen-Avg Task-aware Acc | 0.326 | 0.379 | 0.395 | 0.389 | -0.058 | 0.024 | 0.039 |  |
| instrdialog | Forgetting | 0.128 | 0.083 | 0.078 | 0.089 | 0.056 | -0.017 | -0.022 | -0.056 |
| instrdialog | Task-Aware Forgetting | 0.133 | 0.072 | 0.078 | 0.083 | 0.067 | -0.028 | -0.022 | -0.067 |
| instrdialog | Token F1 | 0.408 | 0.455 | 0.498 | 0.466 | -0.079 | 0.008 | 0.051 |  |
| instrdialog | ROUGE-L | 0.404 | 0.451 | 0.493 | 0.460 | -0.079 | 0.007 | 0.049 |  |
| instrdialog | BLEU | 0.450 | 0.490 | 0.513 | 0.484 | -0.069 | 0.005 | 0.029 |  |
| instrdialog | LCS Overlap | 0.398 | 0.442 | 0.491 | 0.452 | -0.083 | 0.002 | 0.051 |  |
| instrdialog | Current-Seg Acc | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| instrdialog | Current-Seg Task-aware Acc | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| instrdialog | Task-aware Score Mean | 0.326 | 0.379 | 0.395 | 0.389 | -0.058 | 0.024 | 0.039 |  |
| instrdialogpp | Seen-Avg Acc | 0.411 | 0.426 | 0.416 | 0.461 | 0.029 | 0.030 | 0.020 |  |
| instrdialogpp | Seen-Avg Task-aware Acc | 0.411 | 0.429 | 0.421 | 0.468 | 0.029 | 0.033 | 0.025 |  |
| instrdialogpp | Forgetting | 0.124 | 0.095 | 0.103 | 0.062 | -0.011 | -0.035 | -0.027 | 0.011 |
| instrdialogpp | Task-Aware Forgetting | 0.127 | 0.095 | 0.103 | 0.062 | -0.008 | -0.036 | -0.028 | 0.008 |
| instrdialogpp | Token F1 | 0.543 | 0.537 | 0.540 | 0.570 | 0.036 | 0.012 | 0.015 |  |
| instrdialogpp | ROUGE-L | 0.535 | 0.529 | 0.532 | 0.563 | 0.037 | 0.013 | 0.015 |  |
| instrdialogpp | BLEU | 0.580 | 0.569 | 0.581 | 0.598 | 0.028 | 0.003 | 0.015 |  |
| instrdialogpp | LCS Overlap | 0.534 | 0.525 | 0.529 | 0.558 | 0.038 | 0.010 | 0.014 |  |
| instrdialogpp | Current-Seg Acc | 0.200 | 0.000 | 0.200 | 0.000 | 0.000 | -0.200 | 0.000 |  |
| instrdialogpp | Current-Seg Task-aware Acc | 0.200 | 0.000 | 0.200 | 0.000 | 0.000 | -0.200 | 0.000 |  |
| instrdialogpp | Task-aware Score Mean | 0.411 | 0.429 | 0.421 | 0.468 | 0.029 | 0.033 | 0.025 |  |

### seed 789 旧协议（缺格为 —）

| Data | metric | rand_ce | proto_ce | rand_opsd | proto_opsd | interaction | proto main | OPSD main | forgetting sign-flip |
|---|---|---|---|---|---|---|---|---|---|
| instrdialog | Seen-Avg Acc | 0.368 | 0.374 | 0.368 | 0.384 | 0.011 | 0.011 | 0.005 |  |
| instrdialog | Seen-Avg Task-aware Acc | 0.379 | 0.384 | 0.379 | 0.400 | 0.016 | 0.013 | 0.008 |  |
| instrdialog | Forgetting | 0.078 | 0.056 | 0.083 | 0.050 | -0.011 | -0.028 | 0.000 | 0.011 |
| instrdialog | Task-Aware Forgetting | 0.067 | 0.050 | 0.072 | 0.039 | -0.017 | -0.025 | -0.003 | 0.017 |
| instrdialog | Token F1 | 0.493 | 0.480 | 0.480 | 0.496 | 0.028 | 0.001 | 0.002 |  |
| instrdialog | ROUGE-L | 0.489 | 0.475 | 0.476 | 0.491 | 0.029 | 0.000 | 0.002 |  |
| instrdialog | BLEU | 0.504 | 0.501 | 0.491 | 0.515 | 0.027 | 0.011 | 0.000 |  |
| instrdialog | LCS Overlap | 0.493 | 0.477 | 0.480 | 0.491 | 0.027 | -0.003 | 0.001 |  |
| instrdialog | Current-Seg Acc | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| instrdialog | Current-Seg Task-aware Acc | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| instrdialog | Task-aware Score Mean | 0.379 | 0.384 | 0.379 | 0.400 | 0.016 | 0.013 | 0.008 |  |
| instrdialogpp | Seen-Avg Acc | 0.389 | 0.434 | 0.411 | 0.455 | 0.000 | 0.045 | 0.021 |  |
| instrdialogpp | Seen-Avg Task-aware Acc | 0.392 | 0.434 | 0.413 | 0.455 | 0.000 | 0.042 | 0.021 |  |
| instrdialogpp | Forgetting | 0.127 | 0.089 | 0.100 | 0.049 | -0.014 | -0.045 | -0.034 | 0.014 |
| instrdialogpp | Task-Aware Forgetting | 0.127 | 0.089 | 0.100 | 0.049 | -0.014 | -0.045 | -0.034 | 0.014 |
| instrdialogpp | Token F1 | 0.515 | 0.551 | 0.552 | 0.555 | -0.032 | 0.020 | 0.021 |  |
| instrdialogpp | ROUGE-L | 0.508 | 0.544 | 0.543 | 0.550 | -0.030 | 0.022 | 0.020 |  |
| instrdialogpp | BLEU | 0.563 | 0.585 | 0.586 | 0.597 | -0.011 | 0.017 | 0.018 |  |
| instrdialogpp | LCS Overlap | 0.499 | 0.539 | 0.539 | 0.546 | -0.033 | 0.024 | 0.023 |  |
| instrdialogpp | Current-Seg Acc | 0.000 | 0.000 | 0.100 | 0.100 | 0.000 | 0.000 | 0.100 |  |
| instrdialogpp | Current-Seg Task-aware Acc | 0.000 | 0.000 | 0.100 | 0.100 | 0.000 | 0.000 | 0.100 |  |
| instrdialogpp | Task-aware Score Mean | 0.392 | 0.434 | 0.413 | 0.455 | 0.000 | 0.042 | 0.021 |  |
### seed 42 旧协议（缺格为 —）

| Data | metric | rand_ce | proto_ce | rand_opsd | proto_opsd | interaction | proto main | OPSD main | forgetting sign-flip |
|---|---|---|---|---|---|---|---|---|---|
| instrdialog | Seen-Avg Acc | 0.326 | 0.411 | 0.321 | 0.368 | -0.037 | 0.066 | -0.024 |  |
| instrdialog | Seen-Avg Task-aware Acc | 0.332 | 0.426 | 0.332 | 0.379 | -0.047 | 0.071 | -0.024 |  |
| instrdialog | Forgetting | 0.122 | 0.039 | 0.122 | 0.067 | 0.028 | -0.069 | 0.014 | -0.028 |
| instrdialog | Task-Aware Forgetting | 0.122 | 0.022 | 0.117 | 0.056 | 0.039 | -0.081 | 0.014 | -0.039 |
| instrdialog | Token F1 | 0.444 | 0.525 | 0.436 | 0.476 | -0.041 | 0.061 | -0.028 |  |
| instrdialog | ROUGE-L | 0.436 | 0.519 | 0.429 | 0.469 | -0.044 | 0.061 | -0.029 |  |
| instrdialog | BLEU | 0.465 | 0.526 | 0.465 | 0.498 | -0.028 | 0.047 | -0.013 |  |
| instrdialog | LCS Overlap | 0.434 | 0.521 | 0.425 | 0.465 | -0.047 | 0.064 | -0.033 |  |
| instrdialog | Current-Seg Acc | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| instrdialog | Current-Seg Task-aware Acc | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| instrdialog | Task-aware Score Mean | 0.332 | 0.426 | 0.332 | 0.379 | -0.047 | 0.071 | -0.024 |  |
| instrdialogpp | Seen-Avg Acc | 0.416 | 0.453 | 0.400 | 0.442 | 0.005 | 0.039 | -0.013 |  |
| instrdialogpp | Seen-Avg Task-aware Acc | 0.418 | 0.461 | 0.403 | 0.445 | 0.000 | 0.042 | -0.016 |  |
| instrdialogpp | Forgetting | 0.089 | 0.076 | 0.105 | 0.073 | -0.019 | -0.023 | 0.007 | 0.019 |
| instrdialogpp | Task-Aware Forgetting | 0.089 | 0.068 | 0.103 | 0.070 | -0.011 | -0.027 | 0.008 | 0.011 |
| instrdialogpp | Token F1 | 0.526 | 0.569 | 0.508 | 0.564 | 0.013 | 0.049 | -0.011 |  |
| instrdialogpp | ROUGE-L | 0.519 | 0.561 | 0.500 | 0.556 | 0.014 | 0.049 | -0.011 |  |
| instrdialogpp | BLEU | 0.576 | 0.603 | 0.557 | 0.592 | 0.008 | 0.031 | -0.015 |  |
| instrdialogpp | LCS Overlap | 0.519 | 0.562 | 0.498 | 0.559 | 0.017 | 0.052 | -0.012 |  |
| instrdialogpp | Current-Seg Acc | 0.300 | 0.100 | 0.100 | 0.000 | 0.100 | -0.150 | -0.150 |  |
| instrdialogpp | Current-Seg Task-aware Acc | 0.300 | 0.100 | 0.100 | 0.000 | 0.100 | -0.150 | -0.150 |  |
| instrdialogpp | Task-aware Score Mean | 0.418 | 0.461 | 0.403 | 0.445 | 0.000 | 0.042 | -0.016 |  |

## E5 轨迹（来自各段 eval 的 per_segment_accuracy）

- `v3_proto_opsd_instrdialog_seed456`: flat=8, forgotten_then_recovered=2, kept=5, learned_then_forgotten=1, learned_then_forgotten+recovered=2
- `v3_rand_opsd_instrdialog_seed456`: flat=8, forgotten_then_recovered=1, kept=3, learned_then_forgotten=3, learned_then_forgotten+recovered=3
- `v3h_proto_opsd_instrdialog_seed456`: flat=8, forgotten_then_recovered=1, kept=1, learned_then_forgotten=4, learned_then_forgotten+recovered=4

## E8 真实 case（eval_debug 末段，各取一条 match 与一条不 match）

- `v3_proto_opsd_instrdialog_seed456` seg 1 match=True
  - gold: Find information on hip fractures in the elderly.
  - pred: Find information on hip fractures in the elderly.
- `v3_proto_opsd_instrdialog_seed456` seg 0 match=False
  - gold: Yeah, honestly, the more I think about it, the more I think it'll just be easier to wipe turkey off the map really quickly. I've done it in tourneys a few times before and it just 
  - pred: I think that's a good plan.
- `v3h_proto_opsd_instrdialog_seed456` seg 1 match=True
  - gold: How do I get my free annual credit report?
  - pred: How do I get my free annual credit report?
- `v3h_proto_opsd_instrdialog_seed456` seg 0 match=False
  - gold: Yeah, honestly, the more I think about it, the more I think it'll just be easier to wipe turkey off the map really quickly. I've done it in tourneys a few times before and it just 
  - pred: I think that’s a good plan. Bur arranged bounce bounce bounce bounce bounce bounce bounce bounce bounce bounce bounce bounce bounce bounce bounce bounce bounce bounce bounce bounce

## E4 训练标量（metrics.jsonl 均值）

温度保持 `rollout_temperature: 1.0`，JSD 没有单独蒸馏温度。梯度余弦未测。

- `v3_proto_opsd_instrdialog_seed456`: CE 1.226, OPSD 0.150, rollout tokens 9.627, grad norm 5.665, segments logged 19
- `v3_rand_ce_instrdialog_seed456`: CE 1.367, OPSD 0.000, rollout tokens 0.000, grad norm 6.190, segments logged 29
- `v3h_proto_opsd_instrdialog_seed456`: CE 1.252, OPSD 0.153, rollout tokens 9.932, grad norm 6.732, segments logged 19

## E3

顺序 SFT 与段内交替目标已有 v1 run，协议与本表 v3 / v3h 不同，这里只作对照说明，不重训。

## E7 仓库内真实 baseline 入口

- `baselines/advanced_baselines/continual_t0/method.py`
- `baselines/advanced_baselines/lb_cl/method.py`
- `baselines/advanced_baselines/o_lora/method.py`
- `baselines/advanced_baselines/progressive_prompts/method.py`
- `baselines/basic_baselines/ce_opsd/method.py`
- `baselines/basic_baselines/ce_opsd_replay/method.py`
- `baselines/basic_baselines/gold_opsd/method.py`
- `baselines/basic_baselines/metric_opsd/method.py`
- `baselines/basic_baselines/metric_opsd_replay/method.py`
- `baselines/basic_baselines/neg_opsd/method.py`
- `baselines/basic_baselines/opsd/method.py`
- `baselines/basic_baselines/periodic_multilora/method.py`
- `baselines/basic_baselines/replay_lora/method.py`
- `baselines/basic_baselines/router_only/method.py`
- `baselines/basic_baselines/seg_opsd/method.py`
- `baselines/basic_baselines/seg_opsd_replay/method.py`
- `baselines/basic_baselines/sequential_lora/method.py`
- `baselines/citb/adaptercl/method.py`
- `baselines/citb/agem/method.py`
- `baselines/citb/ewc/method.py`
- `baselines/citb/ft_no_instr/method.py`
- `baselines/citb/init/method.py`
- `baselines/citb/l2/method.py`
- `baselines/citb/multi/method.py`
- `baselines/citb/replay/method.py`
- `baselines/clora/method.py`
- `baselines/gainlora/method.py`
- `baselines/gorp/method.py`
- `baselines/inscl/method.py`
- `baselines/kpig/method.py`
- `baselines/lcia/method.py`
- `baselines/onedit/method.py`
- `baselines/saber/method.py`
- `baselines/sapt/method.py`
- `baselines/sdft/method.py`
- `baselines/ssr/method.py`
- `baselines/treelora/method.py`

官方第三方树未纳入本仓库。

## E12 实测时长与显存

| run | start | end | elapsed_sec | gpu_after |
|---|---|---|---|---|
| v3_proto_ce_instrdialog_seed42 | 2026-10-31T14:11:59+08:00 | 2026-10-31T14:19:49+08:00 | 469.5 | 0 MiB, 48509 MiB |
| v3_proto_ce_instrdialog_seed789 | 2026-10-31T14:04:03+08:00 | 2026-10-31T14:11:59+08:00 | 475.8 | 0 MiB, 48509 MiB |
| v3_proto_ce_instrdialogpp_seed42 | 2026-10-31T16:02:44+08:00 | 2026-10-31T16:23:56+08:00 | 1271.9 | 0 MiB, 48509 MiB |
| v3_proto_ce_instrdialogpp_seed789 | 2026-10-31T15:41:25+08:00 | 2026-10-31T16:02:44+08:00 | 1278.8 | 0 MiB, 48509 MiB |
| v3_rand_ce_instrdialog_seed42 | 2026-10-31T13:56:13+08:00 | 2026-10-31T14:04:03+08:00 | 469.3 | 0 MiB, 48509 MiB |
| v3_rand_ce_instrdialog_seed789 | 2026-10-31T13:48:14+08:00 | 2026-10-31T13:56:13+08:00 | 479.0 | 0 MiB, 48509 MiB |
| v3_rand_ce_instrdialogpp_seed42 | 2026-10-31T15:19:30+08:00 | 2026-10-31T15:41:25+08:00 | 1314.4 | 0 MiB, 48509 MiB |
| v3_rand_ce_instrdialogpp_seed789 | 2026-10-31T14:57:29+08:00 | 2026-10-31T15:19:30+08:00 | 1321.0 | 0 MiB, 48509 MiB |
| v3_rand_opsd_instrdialog_seed42 | 2026-10-31T14:39:01+08:00 | 2026-10-31T14:57:28+08:00 | 1107.0 | 0 MiB, 48509 MiB |
| v3_rand_opsd_instrdialog_seed789 | 2026-10-31T14:19:49+08:00 | 2026-10-31T14:39:01+08:00 | 1151.9 | 0 MiB, 48509 MiB |
| v3_rand_opsd_instrdialogpp_seed42 | 2026-10-31T17:06:38+08:00 | 2026-10-31T17:50:33+08:00 | 2635.0 | 0 MiB, 48509 MiB |
| v3_rand_opsd_instrdialogpp_seed789 | 2026-10-31T16:23:57+08:00 | 2026-10-31T17:06:37+08:00 | 2560.7 | 0 MiB, 48509 MiB |
| v3h_proto_ce_instrdialog_seed456 | 2026-10-31T18:00:57+08:00 | 2026-10-31T18:11:41+08:00 | 643.8 | 0 MiB, 48509 MiB |
| v3h_proto_ce_instrdialogpp_seed456 | 2026-10-31T19:25:00+08:00 | 2026-10-31T19:55:30+08:00 | 1829.3 | 0 MiB, 48509 MiB |
| v3h_proto_opsd_instrdialog_seed456 | 2026-10-31T18:32:39+08:00 | 2026-10-31T18:53:26+08:00 | 1246.3 | 0 MiB, 48509 MiB |
| v3h_proto_opsd_instrdialog_seed456_smoke2seg | 2026-10-31T13:38:19+08:00 | 2026-10-31T13:40:50+08:00 | 151.1 | 0 MiB, 48509 MiB |
| v3h_proto_opsd_instrdialogpp_seed456 | 2026-10-31T21:14:30+08:00 | 2026-10-31T22:03:41+08:00 | 2951.2 | 0 MiB, 48509 MiB |
| v3h_rand_ce_instrdialog_seed456 | 2026-10-31T17:50:33+08:00 | 2026-10-31T18:00:57+08:00 | 623.5 | 0 MiB, 48509 MiB |
| v3h_rand_ce_instrdialogpp_seed456 | 2026-10-31T18:53:26+08:00 | 2026-10-31T19:25:00+08:00 | 1893.5 | 0 MiB, 48509 MiB |
| v3h_rand_opsd_instrdialog_seed456 | 2026-10-31T18:11:41+08:00 | 2026-10-31T18:32:39+08:00 | 1257.5 | 0 MiB, 48509 MiB |
| v3h_rand_opsd_instrdialogpp_seed456 | 2026-10-31T20:22:40+08:00 | 2026-10-31T21:14:29+08:00 | 3109.5 | 0 MiB, 48509 MiB |

<!-- BEGIN PENDING_EXT -->
## Pending extensions（seed 456，本轮新跑）

数字只来自 `final_metrics.json`。缺测写 —。JSD 无单独蒸馏温度；`rollout_temperature` 保持 1.0。
E3 SFT→OPSD：Blocked（CITB Sequential adapter 走非 multiref 流且快照无 format_normalize，不是合法 v3 协议 init）。
不重跑 20% 随机 / 全相关 / 全退化 / 10+10 历史格，也不开 50–90% replay 或新的 789/42。

### E2 15/5 与 5/15 槽混合（train_holdout）

| run | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3h_proto_opsd_n15f5_instrdialog_seed456 | 0.363 | 0.363 | 0.078 | 0.078 | 0.460 | 0.454 | 0.482 | 0.452 | 0.000 | 0.000 | 0.363 |
| v3h_proto_opsd_n5f15_instrdialog_seed456 | 0.411 | 0.411 | 0.044 | 0.044 | 0.507 | 0.502 | 0.524 | 0.494 | 0.000 | 0.000 | 0.411 |
| v3h_proto_opsd_n15f5_instrdialogpp_seed456 | 0.400 | 0.403 | 0.108 | 0.108 | 0.513 | 0.507 | 0.562 | 0.504 | 0.000 | 0.000 | 0.403 |
| v3h_proto_opsd_n5f15_instrdialogpp_seed456 | 0.426 | 0.429 | 0.084 | 0.084 | 0.544 | 0.536 | 0.595 | 0.531 | 0.000 | 0.000 | 0.429 |

| run | start | end | elapsed_sec | gpu_after |
|---|---|---|---|---|
| v3h_proto_opsd_n15f5_instrdialog_seed456 | 2026-11-01T03:39:19+08:00 | 2026-11-01T03:59:13+08:00 | 1193.2 | 0 MiB, 48509 MiB |
| v3h_proto_opsd_n5f15_instrdialog_seed456 | 2026-11-01T03:59:13+08:00 | 2026-11-01T04:19:49+08:00 | 1236.4 | 0 MiB, 48509 MiB |
| v3h_proto_opsd_n15f5_instrdialogpp_seed456 | 2026-11-01T04:19:50+08:00 | 2026-11-01T05:07:59+08:00 | 2889.5 | 0 MiB, 48509 MiB |
| v3h_proto_opsd_n5f15_instrdialogpp_seed456 | 2026-11-01T05:08:00+08:00 | 2026-11-01T05:58:21+08:00 | 3021.2 | 0 MiB, 48509 MiB |

### E4 λ ∈ {0.05, 0.10, 0.50}（PGDR+CE+OPSD holdout）

| run | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3h_proto_opsd_lam005_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_lam010_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_lam050_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_lam005_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_lam010_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_lam050_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |

| run | start | end | elapsed_sec | gpu_after |
|---|---|---|---|---|
| v3h_proto_opsd_lam005_instrdialog_seed456 | 2026-11-01T06:01:49+08:00 | — | — | — |
| v3h_proto_opsd_lam010_instrdialog_seed456 | — | — | — | — |
| v3h_proto_opsd_lam050_instrdialog_seed456 | — | — | — | — |
| v3h_proto_opsd_lam005_instrdialogpp_seed456 | — | — | — | — |
| v3h_proto_opsd_lam010_instrdialogpp_seed456 | — | — | — | — |
| v3h_proto_opsd_lam050_instrdialogpp_seed456 | — | — | — | — |

### E4 off-policy gold tokens vs on-policy（λ=0.2 holdout）

| run | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3h_proto_opsd_offpol_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_offpol_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |

| run | start | end | elapsed_sec | gpu_after |
|---|---|---|---|---|
| v3h_proto_opsd_offpol_instrdialog_seed456 | — | — | — | — |
| v3h_proto_opsd_offpol_instrdialogpp_seed456 | — | — | — | — |

### E6 replay_ratio ∈ {0, 0.10, 0.30, 0.40}

| run | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3_proto_opsd_rr0_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_rr10_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_rr30_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_rr40_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3_proto_opsd_rr0_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_rr10_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_rr30_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_rr40_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |

| run | start | end | elapsed_sec | gpu_after |
|---|---|---|---|---|
| v3_proto_opsd_rr0_instrdialog_seed456 | — | — | — | — |
| v3h_proto_opsd_rr10_instrdialog_seed456 | — | — | — | — |
| v3h_proto_opsd_rr30_instrdialog_seed456 | — | — | — | — |
| v3h_proto_opsd_rr40_instrdialog_seed456 | — | — | — | — |
| v3_proto_opsd_rr0_instrdialogpp_seed456 | — | — | — | — |
| v3h_proto_opsd_rr10_instrdialogpp_seed456 | — | — | — | — |
| v3h_proto_opsd_rr30_instrdialogpp_seed456 | — | — | — | — |
| v3h_proto_opsd_rr40_instrdialogpp_seed456 | — | — | — | — |

### E3 v3 协议对照（direct OPSD / seg_opsd K=25）

| run | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3_opsd_direct_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3_opsd_direct_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3_seg_opsd_k25_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3_seg_opsd_k25_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |

| run | start | end | elapsed_sec | gpu_after |
|---|---|---|---|---|
| v3_opsd_direct_instrdialog_seed456 | — | — | — | — |
| v3_opsd_direct_instrdialogpp_seed456 | — | — | — | — |
| v3_seg_opsd_k25_instrdialog_seed456 | — | — | — | — |
| v3_seg_opsd_k25_instrdialogpp_seed456 | — | — | — | — |

### E9 buffer_k ∈ {5, 20, 50}

| run | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3h_proto_opsd_bufk5_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_bufk20_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_bufk50_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_bufk5_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_bufk20_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_bufk50_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |

| run | start | end | elapsed_sec | gpu_after |
|---|---|---|---|---|
| v3h_proto_opsd_bufk5_instrdialog_seed456 | — | — | — | — |
| v3h_proto_opsd_bufk20_instrdialog_seed456 | — | — | — | — |
| v3h_proto_opsd_bufk50_instrdialog_seed456 | — | — | — | — |
| v3h_proto_opsd_bufk5_instrdialogpp_seed456 | — | — | — | — |
| v3h_proto_opsd_bufk20_instrdialogpp_seed456 | — | — | — | — |
| v3h_proto_opsd_bufk50_instrdialogpp_seed456 | — | — | — | — |

### E10 InstrDialog 任务顺序置换

| run | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3h_proto_opsd_ordrev_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_ordshuf_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |

| run | start | end | elapsed_sec | gpu_after |
|---|---|---|---|---|
| v3h_proto_opsd_ordrev_instrdialog_seed456 | — | — | — | — |
| v3h_proto_opsd_ordshuf_instrdialog_seed456 | — | — | — | — |

### E11 holdout 全相关 / 全退化 20%

| run | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3h_proto_opsd_allnb_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_allfg_instrdialog_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_allnb_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |
| v3h_proto_opsd_allfg_instrdialogpp_seed456 | — | — | — | — | — | — | — | — | — | — | — |

| run | start | end | elapsed_sec | gpu_after |
|---|---|---|---|---|
| v3h_proto_opsd_allnb_instrdialog_seed456 | — | — | — | — |
| v3h_proto_opsd_allfg_instrdialog_seed456 | — | — | — | — |
| v3h_proto_opsd_allnb_instrdialogpp_seed456 | — | — | — | — |
| v3h_proto_opsd_allfg_instrdialogpp_seed456 | — | — | — | — |
<!-- END PENDING_EXT -->
