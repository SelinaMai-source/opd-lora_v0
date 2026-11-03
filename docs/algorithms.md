# 算法

这里只记录仓库里已经实现、并且 v3c 实际调用的定义。

## 旧配额

`methods.pgdr.quota(n, ratio) = max(0, int(round(n * ratio)))`，用的是 Python 的银行家舍入。

- `uniform`：`quota(45, 0.2) = 9`
- `proto`：`quota(45, 0.1) + quota(45, 0.1) = 4 + 4 = 8`

两条路径的回放条数不同。不能把它们的均值和 v3c 放在一起。

## 修正配额

`core.replay_budget.mixed_batch_n_replay`：

`n_replay = half_up(ρ / (1 - ρ) × n_current)`，ρ = 0.20。

`half_up` 对非负数是远离零的四舍五入，不是银行家舍入。45 条 current 得到 11。实现比例 `11 / (11 + 45) = 11/56`。日志写 `realized_replay_fraction`，不写成严格 20%。

Related 与 Degraded 用最大余数法，和等于 `n_replay`。余数相等时 Related 先拿，所以 11 分成 6+5。R、U、P 都拿这同一个 11。历史池不够就记 shortfall，不再各自 `round`。训练 seed 不改变目标条数。

## 选择

- R：历史记忆均匀抽 `n_replay`。
- U：Related 只做格式兼容过滤，然后均匀抽。余弦不改变入选概率。
- P：兼容段按显式余弦降序。余弦是两侧先 L2 归一化再点积。并列按 `segment_id` 升序。从高到低的段里抽该段 K=10 记忆，直到 Related 配额填满。不够再从其余合法记忆补，并记 `fallback_count`。

Degraded 只读 train-holdout 遗忘。不读未来段，不读官方 test，不用 gold 做排序。

## OPSD

同一步损失是 `L = L_CE + λ L_OPSD`。`λ <= 0` 时不加入 OPSD。散度是 `methods.opsd.token_divergence` 里的 JSD 或 KL。教师前向在 `disable_adapter()` 下、`torch.no_grad` 中计算。本文件不另写一套温度或损失。
