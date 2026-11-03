# 旧协议与修正协议

两套数字不能相减，也不能放进同一个 Mean±Std。

| | 旧 `uniform` / `proto`（含 v3h） | v3c |
|---|---|---|
| 45 条 current 的回放条数 | uniform 9；proto 4+4=8 | R、U、P 都是 11 |
| 舍入 | Python `round` | `half_up`，再最大余数 |
| Related | proto 先点积排序，再把兼容段拼成一个池均匀抽 | U 仍是兼容后均匀抽；P 按显式 L2 余弦从高到低填槽 |
| 余弦 | `_cosine` 是未归一化点积；激活若已经单位化，点积在数值上等于余弦，但默认近邻槽不用它抽样 | P 在打分前对两侧再做 L2 归一化，并用这个顺序填 Related |
| 退化槽 | 旧 `eval_split_monitor` 可读官方 eval 遗忘 | v3c 只读 train-holdout，`holdout_seed=456` |
| 配置开关 | 旧 yaml 不写 `budget_policy: unified` | 缺 `monitor=train_holdout`、`holdout_seed=456` 或 `budget_policy=unified` 时队列拒绝启动 |

`v3h_*` 目录不覆盖，`final_metrics.json` 不改写。v3c 的 run 名以 `v3c_` 开头，结果在 `docs/ACL_v3_Corrected_Method_Results.md`。

Protocol A 是更早的评测或选择协议，同样不进入 v3c 的三种子均值。
