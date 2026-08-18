## Baseline：Replay LoRA

### 核心思想

- 只有**一个** LoRA 分支（adapter）。
- 维护一个跨 segment 的 **replay buffer**。
- 每个 segment 训练时，将“当前段样本”与“buffer 中采样的历史样本”混合训练，缓解遗忘。

### 与我们方法的差异

我们的方法通过漂移检测触发分支生成，并用 bank + router 做分支专门化与选择，外加 anti-overlap 正则；Replay LoRA 不创建分支，也不做路由，而是依赖 replay 机制来减少遗忘。

### 当前代码中的简化点

- replay 策略目前为简单的 uniform sampling + FIFO 截断（未来可扩展 reservoir / task-aware）。
- debug 模式训练为记忆映射，用于验证“混合训练集/缓解遗忘”的调度逻辑是否连通。

