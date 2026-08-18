## Baseline：Router-only

### 核心思想

- 预先准备多个 LoRA 分支（固定数量或按预定义计划生成）。
- 使用 **router** 对每个样本进行分支选择（hard routing）。
- 不使用漂移检测：分支集合不根据漂移事件动态扩展（与我们方法不同）。

### 与我们方法的差异

我们的方法是“漂移触发 + bank 管理 + router 选择 + overlap 正则”一体化：
- drift detector 决定何时需要新分支
- bank 负责生成/冻结/限制分支数
- router 做样本级分支选择
- overlap loss 鼓励分支表征差异化

Router-only baseline 只保留 router 选择分支，不引入漂移触发与 anti-overlap 机制。

### 当前代码中的简化点

- router 当前是启发式打分 + warmup 逻辑，提供了未来 pseudo-label 训练的接口位置。
- debug 模式下分支训练只是切换 adapter 名称并更新 debug 模型的记忆映射，用于验证“路由-分支切换-训练”这条链路能跑通。

