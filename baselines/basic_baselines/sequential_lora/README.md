## Baseline：Sequential LoRA

### 核心思想

- 只有**一个** LoRA 分支（adapter）。
- 按 segment 顺序持续训练：第 \(t\) 段训练完成后直接进入第 \(t+1\) 段。
- 不使用漂移检测、不使用 replay、不使用路由。

### 与我们方法的差异

我们的方法会在检测到分布漂移后（或按策略）生成新的 LoRA 分支，并通过 router 在推理/训练时选择合适分支，同时加入 anti-overlap 正则鼓励分支表征多样性；Sequential LoRA 完全没有这些机制。

### 当前代码中的简化点

- 该仓库的 debug 模式使用 `DebugTextModel`，训练等价于记忆映射，用于验证管线正确性。
- 在真实 HF + PEFT 实现中，LoRA 的训练将由 torch optimizer 与 PEFT adapter 参数驱动；该 baseline 的接口与调度逻辑保持不变。

