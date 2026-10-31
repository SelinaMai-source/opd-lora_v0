# 已有 run 清单

字段级清单是 `results/manifest.csv`。每一行只使用该 run 自己的 `config_snapshot.yaml`、`final_metrics.json`、`metrics.jsonl` 和 `run_manifest.json`。快照里没有的字段写 `null`。`training_steps` 是 `metrics.jsonl` 里每个 `segment_id` 最后一条的 `train.batches` 之和。v3 因子格没有记录 token 数，所以 `training_tokens` 为 `null`；没有把 Lit run 的 token 数抄过来。

扫描范围是 `results/runs` 下带 `final_metrics.json` 的 `v3_*`、`lit_*`、`v2pre_citb_*`、`v2pre_lcia_*`，共 110 行：v3 41、Lit 33、CITB 33、LCIA 3。`_collapsed_bak_20261003` 不在这层目录里，没有计入。`lit_kpig_instrdialogpp_seed456_failed_oom_seg27` 没有 `final_metrics.json`，没有计入。

## 十二个已完成因子格

四个方法、seed 456、两个 benchmark，加上 `v3_proto_opsd` 的 seed 789 和 42。计划里的 Acc/F 参考值与 `final.eval` 四舍五入到三位小数后全部一致。下表三位小数来自实测，不是把参考值再抄一遍。

| run | Acc | F | 对照 |
|---|---:|---:|---|
| `v3_rand_ce_instrdialog_seed456` | 0.395 | 0.072 | 参考 0.395 / 0.072 |
| `v3_proto_ce_instrdialog_seed456` | 0.384 | 0.050 | 参考 0.384 / 0.050 |
| `v3_rand_opsd_instrdialog_seed456` | 0.400 | 0.044 | 参考 0.400 / 0.044 |
| `v3_proto_opsd_instrdialog_seed456` | 0.432 | 0.017 | 参考 0.432 / 0.017 |
| `v3_rand_ce_instrdialogpp_seed456` | 0.376 | 0.130 | 参考 0.376 / 0.130 |
| `v3_proto_ce_instrdialogpp_seed456` | 0.474 | 0.035 | 参考 0.474 / 0.035 |
| `v3_rand_opsd_instrdialogpp_seed456` | 0.389 | 0.116 | 参考 0.389 / 0.116 |
| `v3_proto_opsd_instrdialogpp_seed456` | 0.450 | 0.068 | 参考 0.450 / 0.068 |
| `v3_proto_opsd_instrdialog_seed789` | 0.384 | 0.050 | 参考 0.384 / 0.050 |
| `v3_proto_opsd_instrdialog_seed42` | 0.368 | 0.067 | 参考 0.368 / 0.067 |
| `v3_proto_opsd_instrdialogpp_seed789` | 0.455 | 0.049 | 参考 0.455 / 0.049 |
| `v3_proto_opsd_instrdialogpp_seed42` | 0.442 | 0.073 | 参考 0.442 / 0.073 |

这 12 个快照的共同设置：Llama-3.1-8B-Instruct、LoRA r=16、α=32、`q_proj|v_proj`、lr=0.0002、batch 2、每段 1 epoch、`buffer_k=10`、`replay_ratio=0.2`、流文件为对应 benchmark 的 `*_multiref.json`、`format_normalize=true`。proto 格是 `strategy=proto`、近邻比 0.1、退化比 0.1。随机格是 `strategy=uniform`。CE 格 `lambda_opsd=0`，OPSD 格为 0.2。OPSD 的 rollout 温度为 1.0，JSD β=0.5，教师格式约束为 true。

原始末点（未四舍五入）：

- InstrDialog seed 456：随机 CE 0.394737 / 0.072222；PGDR+CE 0.384211 / 0.050000；随机+OPSD 0.400000 / 0.044444；PGDR+OPSD 0.431579 / 0.016667。
- InstrDialog++ seed 456：0.376316 / 0.129730；0.473684 / 0.035135；0.389474 / 0.116216；0.450000 / 0.067568。
- PGDR+OPSD seed 789：InstrDialog 0.384211 / 0.050000；InstrDialog++ 0.455263 / 0.048649。
- PGDR+OPSD seed 42：InstrDialog 0.368421 / 0.066667；InstrDialog++ 0.442105 / 0.072973。

seed 789 的 InstrDialog 在三位小数上与 seed 456 的 PGDR+CE 同为 0.384 / 0.050，但是两个目录的原始浮点并不相同：PGDR+CE 的 Acc/F 是 0.38421052631578945 / 0.05，seed 789 的 PGDR+OPSD 是 0.3842105263157894 / 0.049999999999999996。它们各自来自自己的 `final_metrics.json`。

PGDR+OPSD 三个 seed 的样本标准差（n−1）：InstrDialog Acc 0.394737±0.032868，F 0.044444±0.025459；InstrDialog++ Acc 0.449123±0.006623，F 0.063063±0.012772。另外三个方法没有 seed 789/42 的 `final_metrics.json`。缺的 12 个目录是 `v3_{rand_ce,proto_ce,rand_opsd}_{instrdialog,instrdialogpp}_seed{789,42}`。

全流程 step：InstrDialog 四个 seed 456 格以及两个额外 proto seed 都是 565；InstrDialog++ 都是 1135。`training_tokens` 为 `null`。`run_manifest` 的墙钟（秒）与 `logs/*.start`、`logs/*.end` 同一次运行相差约 3–5 秒。InstrDialog CE 约 372–388 秒，CE+OPSD 约 877–961 秒；InstrDialog++ CE 约 1089–1119 秒，CE+OPSD 约 1995–2103 秒。日志里没有峰值显存，这里不写显存。

`v3_rand_ce_instrdialog_seed456` 的 jsonl 含段 0–9 的前缀。step 565 已经丢掉前缀。见审计 AUD-P2-003。

## 协议分组

- 同评测协议的全流程（`format_normalize=true`、multiref 流、完整段数、11 个指标都在）：46 行，在 `results/tables/eleven_metrics_comparable.csv`。其中 proto 且 `forgotten_ratio>0` 的行仍受 AUD-P0-001 约束，不能改标成训练集留出。
- CITB/LCIA：26 个全流程在 `results/tables/eleven_metrics_citb_lcia.csv`。流文件不是 multiref，训练行数更多，快照没有 `format_normalize` 键。见 AUD-P1-003。
- 冒烟与单段 isolation 不进上述两张全流程表。15 个 `v3_transfer_iso_*` 是单段，且从已有 proto adapter 接着训。
- 9 个 Lit 冒烟占位见 AUD-P1-002。

均匀随机回放不调用 `_select_forgotten`。全近邻（`forgotten_ratio: 0`）也不调用。这两类不标 `eval_split_monitor`。

## 表

- `results/tables/factorial_interaction_seed456.csv`：seed 456 的 2×2，11 个指标。交互是 `Y(PGDR+OPSD) − Y(PGDR+CE) − Y(Random+OPSD) + Y(Random+CE)`，用未四舍五入的 `final.eval`。Forgetting 的 `interaction_raw` 仍是原始遗忘（越小越好）。`interaction_higher_is_better` 只在两个遗忘指标上取相反数。
- `results/tables/proto_opsd_seeds.csv`：已有的 6 个 PGDR+OPSD seed。
- `results/tables/allocation_existing_seed456.csv`：已有的 20% 随机、20% 近邻、10%+10%、20% 退化，都是 λ=0.2。没有 15/5 和 5/15。
- `results/tables/replay_selector_audit.csv`、`results/tables/wall_clock.csv`。

## 补跑（2026-10-31）

审计写这份清单时，seed 789/42 的 `v3_{rand_ce,proto_ce,rand_opsd}` 十二个目录还不存在。这十二个旧协议 run 与八个 seed 456 的 `v3h_` 全流程 run 现已完成，产物在 `/root/autodl-tmp/rp_lora_v3_runs/runs/<run_id>/final_metrics.json`。上文十二格表与原始末点保持审计当时的实测，不改写。本节数字只来自补跑目录自己的 `final.eval`。

旧协议补跑使用 `eval_split_monitor`。`v3h_` 使用 `train_holdout`（每段 50 条训练里留出 5 条，seed 456 的四个方法共用同一划分），不读取官方 eval 做选样。冒烟 `v3h_proto_opsd_instrdialog_seed456_smoke2seg` 审计为 SMOKE_HOLDOUT_OK，只跑 2 段，不进入 `eleven_metrics_comparable.csv`。

| run | monitor | Acc | F | segments |
|---|---|---:|---:|---:|
| `v3_proto_ce_instrdialog_seed42` | eval_split_monitor | 0.411 | 0.039 | 19 |
| `v3_proto_ce_instrdialog_seed789` | eval_split_monitor | 0.374 | 0.056 | 19 |
| `v3_proto_ce_instrdialogpp_seed42` | eval_split_monitor | 0.453 | 0.076 | 38 |
| `v3_proto_ce_instrdialogpp_seed789` | eval_split_monitor | 0.434 | 0.089 | 38 |
| `v3_rand_ce_instrdialog_seed42` | eval_split_monitor | 0.326 | 0.122 | 19 |
| `v3_rand_ce_instrdialog_seed789` | eval_split_monitor | 0.368 | 0.078 | 19 |
| `v3_rand_ce_instrdialogpp_seed42` | eval_split_monitor | 0.416 | 0.089 | 38 |
| `v3_rand_ce_instrdialogpp_seed789` | eval_split_monitor | 0.389 | 0.127 | 38 |
| `v3_rand_opsd_instrdialog_seed42` | eval_split_monitor | 0.321 | 0.122 | 19 |
| `v3_rand_opsd_instrdialog_seed789` | eval_split_monitor | 0.368 | 0.083 | 19 |
| `v3_rand_opsd_instrdialogpp_seed42` | eval_split_monitor | 0.400 | 0.105 | 38 |
| `v3_rand_opsd_instrdialogpp_seed789` | eval_split_monitor | 0.411 | 0.100 | 38 |
| `v3h_proto_ce_instrdialog_seed456` | train_holdout | 0.363 | 0.083 | 19 |
| `v3h_proto_ce_instrdialogpp_seed456` | train_holdout | 0.426 | 0.095 | 38 |
| `v3h_proto_opsd_instrdialog_seed456` | train_holdout | 0.384 | 0.089 | 19 |
| `v3h_proto_opsd_instrdialogpp_seed456` | train_holdout | 0.461 | 0.062 | 38 |
| `v3h_rand_ce_instrdialog_seed456` | train_holdout | 0.321 | 0.128 | 19 |
| `v3h_rand_ce_instrdialogpp_seed456` | train_holdout | 0.411 | 0.124 | 38 |
| `v3h_rand_opsd_instrdialog_seed456` | train_holdout | 0.395 | 0.078 | 19 |
| `v3h_rand_opsd_instrdialogpp_seed456` | train_holdout | 0.416 | 0.103 | 38 |

冒烟末点：Acc 0.000，F 0.000，Token F1 0.309，段数 2。`v3h_rand_opsd_instrdialogpp_seed456` 的 `config_snapshot.yaml` 在开跑后曾被移到同名 `.aborted_before_segment0` 目录，训练继续写回原路径并写出 `final_metrics.json`；快照已按该次启动时写入的文件拷回原 run 目录，指标文件未改。

