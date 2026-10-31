# ABLATION v3 Run Queue

单卡顺序。已有 `final_metrics.json` 且配置指纹一致则跳过。目录已存在但不完整则拒绝覆盖，续跑命令在 `/root/autodl-tmp/rp_lora_v3_runs/logs/<run>.resume_cmd`。

结果目录：`/root/autodl-tmp/rp_lora_v3_runs/runs`。

## 冒烟

- `v3h_proto_opsd_instrdialog_seed456_smoke2seg`：PGDR+CE+OPSD，InstrDialog，`max_segments: 2`，seed 456，`train_holdout`。

## 旧协议 12（eval_split_monitor，seed 789 与 42）

`v3_rand_ce`、`v3_proto_ce`、`v3_rand_opsd` × InstrDialog / InstrDialog++。不重跑 seed 456 因子实验，不重跑已有 `v3_proto_opsd` 的 456/789/42。

## v3h 8（train_holdout，seed 456）

`v3h_{rand_ce,proto_ce,rand_opsd,proto_opsd}` × 两个数据集。冒烟若泄漏则本段不跑。

## 不做

不重跑 CITB / LCIA / v0 / v1 / v2 / Lit。
