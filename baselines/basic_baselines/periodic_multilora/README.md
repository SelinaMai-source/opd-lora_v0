## Baseline：Periodic MultiLoRA

### 核心思想

- 维护多个 LoRA 分支（adapters）。
- **按固定间隔**（每 \(N\) 个 segments）生成一个新分支，并切换到该分支继续训练。
- 不使用漂移检测，不使用 router。
- 推理/评估时（简化）使用**最新分支**。

### 与我们方法的差异

我们的方法是“**事件触发 + 选择性路由**”：
- 漂移检测决定何时 spawn
- bank 负责分支管理与冻结策略
- router 为每个样本选择分支
- overlap loss 鼓励分支表征多样性

Periodic MultiLoRA 则是纯时间表策略：不判断漂移、不做样本级路由。

### 当前代码中的简化点

- 分支上限达到 `max_branches` 后不做删除/合并策略（未来可实现淘汰/合并）。
- debug 模型不是真正的 transformer/LoRA，但分支调度逻辑与统一管线接口是对齐的。

