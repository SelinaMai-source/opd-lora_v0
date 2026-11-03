# 方法

v3c 比较三种回放选择器。它们在同一次 mixed-batch 里拿到同一个回放条数，再和当前段样本拼成一个 batch。当前段有 45 条时，目标回放是 11 条，batch 为 56，实现比例是 11/56。

- **R / random**：从全部历史记忆里均匀抽这 11 条。不用相似度，也不用退化分。
- **U / compat_uniform**：Related 槽在格式兼容的历史记忆里均匀抽。Degraded 槽按 train-holdout 遗忘选段。U 不是余弦原型。
- **P / true_prototype**：Related 槽用已经缓存的冻结 instruction 向量做显式 L2 余弦，从高到低填满配额，段内再抽该段记忆。并列时 `segment_id` 小的在前。Degraded 槽与 U 相同。

段向量不是 gold，不是测试集，也不是样本均值。它是 `disable_adapter()` 之后对该段 instruction 字符串取一次激活。余弦在使用前对两侧再做 L2 归一化。

目标有两种。`λ=0` 时只有 CE。`λ=0.2` 时损失是 CE + λ·JSD。教师是冻结底座，学生是当前 LoRA 的 on-policy rollout。公式在 `methods/opsd.py`，训练时仍由原来的 `CeOpsdMethod` 计算。

遗忘在训练集留出的 5 条上估计，`holdout_seed` 固定 456。官方 eval 只负责报告指标，不参与选样。

旧的 `uniform` / `proto` 仍走 Python `round` 的 9 对 8 配额，供 `v3h_*` 追溯。新实验不调用那条路径。
