# P0 seed 456 只读核对

核对日期：2026-11-02。只读，未改 run 目录，未启动 GPU。

范围：`/root/autodl-tmp/rp_lora_v3_runs/runs` 下 8 个已完成 `v3h_*_seed456`。11 列对照 `docs/ABLATION_V3_实验结果与待补表.md` 小节「11 指标：v3h 训练集留出监控 seed 456」，取 `final_metrics.json` 的 `final.eval.*` 原文后格式化为三位小数，不从 Acc/F 回推其余列。

**结论：全部可复用，不重跑。** 八个均为 Verified Completed。

| Run ID | Status | Config 要点 | 11 指标是否齐 | 泄漏字段 | 段数 | 结果路径 | 失败原因 |
|---|---|---|---|---|---|---|---|
| v3h_rand_ce_instrdialog_seed456 | Verified Completed | train_holdout；holdout_n=5；holdout_seed=456；seed=456；LoRA r=16 alpha=32 q_proj/v_proj；lr=2e-4；buffer_k=10；replay_ratio=0.2；uniform；lambda_opsd=0；benchmark/instrdialog/*_multiref.json；format_normalize=true | 齐，11 列与文档一致（Acc 0.321，F 0.128） | 全段 holdout_in_train=false；eval_split_read_for_selection=false；on_eval_end_ignored=true；CE：opsd_skipped_lambda=true，opsd_applied=false | 19（000–018，末段 eval 存在） | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_ce_instrdialog_seed456 | — |
| v3h_proto_ce_instrdialog_seed456 | Verified Completed | 同上监控/种子/LoRA/lr/buffer/比例/格式归一；strategy=proto；lambda_opsd=0；neighbor_ratio=0.1；forgotten_ratio=0.1；benchmark/instrdialog/*_multiref.json | 齐，11 列与文档一致（Acc 0.363，F 0.083） | 全段 holdout_in_train=false；eval_split_read_for_selection=false；on_eval_end_ignored=true；CE：opsd_skipped_lambda=true，opsd_applied=false | 19（000–018，末段 eval 存在） | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_ce_instrdialog_seed456 | — |
| v3h_rand_opsd_instrdialog_seed456 | Verified Completed | 同上监控/种子/LoRA/lr/buffer/比例/格式归一；strategy=uniform；lambda_opsd=0.2；benchmark/instrdialog/*_multiref.json | 齐，11 列与文档一致（Acc 0.395，F 0.078） | 全段 holdout_in_train=false；eval_split_read_for_selection=false；on_eval_end_ignored=true | 19（000–018，末段 eval 存在） | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_opsd_instrdialog_seed456 | — |
| v3h_proto_opsd_instrdialog_seed456 | Verified Completed | 同上监控/种子/LoRA/lr/buffer/比例/格式归一；strategy=proto；lambda_opsd=0.2；neighbor_ratio=0.1；forgotten_ratio=0.1；benchmark/instrdialog/*_multiref.json | 齐，11 列与文档一致（Acc 0.384，F 0.089） | 全段 holdout_in_train=false；eval_split_read_for_selection=false；on_eval_end_ignored=true | 19（000–018，末段 eval 存在） | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_opsd_instrdialog_seed456 | — |
| v3h_rand_ce_instrdialogpp_seed456 | Verified Completed | 同上监控/种子/LoRA/lr/buffer/比例/格式归一；strategy=uniform；lambda_opsd=0；benchmark/instrdialogpp/*_multiref.json | 齐，11 列与文档一致（Acc 0.411，F 0.124） | 全段 holdout_in_train=false；eval_split_read_for_selection=false；on_eval_end_ignored=true；CE：opsd_skipped_lambda=true，opsd_applied=false | 38（000–037，末段 eval 存在） | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_ce_instrdialogpp_seed456 | — |
| v3h_proto_ce_instrdialogpp_seed456 | Verified Completed | 同上监控/种子/LoRA/lr/buffer/比例/格式归一；strategy=proto；lambda_opsd=0；neighbor_ratio=0.1；forgotten_ratio=0.1；benchmark/instrdialogpp/*_multiref.json | 齐，11 列与文档一致（Acc 0.426，F 0.095） | 全段 holdout_in_train=false；eval_split_read_for_selection=false；on_eval_end_ignored=true；CE：opsd_skipped_lambda=true，opsd_applied=false | 38（000–037，末段 eval 存在） | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_ce_instrdialogpp_seed456 | — |
| v3h_rand_opsd_instrdialogpp_seed456 | Verified Completed | 同上监控/种子/LoRA/lr/buffer/比例/格式归一；strategy=uniform；lambda_opsd=0.2；benchmark/instrdialogpp/*_multiref.json | 齐，11 列与文档一致（Acc 0.416，F 0.103） | 全段 holdout_in_train=false；eval_split_read_for_selection=false；on_eval_end_ignored=true | 38（000–037，末段 eval 存在） | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_rand_opsd_instrdialogpp_seed456 | — |
| v3h_proto_opsd_instrdialogpp_seed456 | Verified Completed | 同上监控/种子/LoRA/lr/buffer/比例/格式归一；strategy=proto；lambda_opsd=0.2；neighbor_ratio=0.1；forgotten_ratio=0.1；benchmark/instrdialogpp/*_multiref.json | 齐，11 列与文档一致（Acc 0.461，F 0.062） | 全段 holdout_in_train=false；eval_split_read_for_selection=false；on_eval_end_ignored=true | 38（000–037，末段 eval 存在） | /root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_opsd_instrdialogpp_seed456 | — |

## 回放计数

`replay_log.jsonl` 全段 `n_current=45`。段 0 的 `n_replay=0`（尚无历史）。段 0 之后：

- uniform（四个 rand）：每段 `n_replay=9`，`n_neighbor=0`，`n_forgotten=0`，`n_uniform=9`。
- proto（四个 proto）：每段 `n_replay=8`，`n_neighbor=4`，`n_forgotten=4`。

四个 CE-only（`*_ce_*`）全程 `opsd_skipped_lambda=true`、`opsd_applied=false`。四个 OPSD 全程 `lambda_opsd=0.2`、`opsd_applied=true`、`opsd_skipped_lambda=false`。每段 `holdout_in_memory=false`，`fit_n=45`。

## Git、checkpoint、壁钟

八个 `run_manifest.json` 的 `git_commit_sha` 与 `git_branch_name` 均为空，`git_is_dirty=false`，`git_status_porcelain=[]`，`status=completed`。快照里没有 commit。

八个都有 `final_adapter/adapter_model.safetensors`（27,280,152 bytes）和 `adapter_config.json`（r=16，lora_alpha=32，target 为 q_proj 与 v_proj）。

壁钟取 `logs/acl_queue.steps.tsv` 中该 run 第一条 `OK` 且带 `sec=` 的行：

| Run ID | 队列时间 | sec |
|---|---|---|
| v3h_rand_ce_instrdialog_seed456 | 2026-10-31T18:00:57+08:00 | 624 |
| v3h_proto_ce_instrdialog_seed456 | 2026-10-31T18:11:41+08:00 | 644 |
| v3h_rand_opsd_instrdialog_seed456 | 2026-10-31T18:32:39+08:00 | 1257 |
| v3h_proto_opsd_instrdialog_seed456 | 2026-10-31T18:53:26+08:00 | 1246 |
| v3h_rand_ce_instrdialogpp_seed456 | 2026-10-31T19:25:00+08:00 | 1894 |
| v3h_proto_ce_instrdialogpp_seed456 | 2026-10-31T19:55:30+08:00 | 1829 |
| v3h_rand_opsd_instrdialogpp_seed456 | 2026-10-31T21:14:29+08:00 | 3110 |
| v3h_proto_opsd_instrdialogpp_seed456 | 2026-10-31T22:03:41+08:00 | 2951 |

数据文件均存在：`benchmark/instrdialog/citb_cl_dialogue_tasks_train50_eval10_multiref.json`，`benchmark/instrdialogpp/citb_cl_38_random_tasks_train50_eval10_multiref.json`。
