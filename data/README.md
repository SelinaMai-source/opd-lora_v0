# 数据

本仓库不提交数据集，也不提交模型权重。

InstrDialog 与 InstrDialog++ 来自 CITB（Zhang et al., Findings of EMNLP 2023，arXiv:2310.14510）。任务文本来自 Super-NaturalInstructions（arXiv:2204.07705）。本地 CITB 检出没有 `LICENSE` 文件。本仓库不替这些数据另选许可证，也不把第三方数据重新许可。

评测使用的是 multiref 流：同一条 `(instruction, input)` 的多条 gold 留在 `outputs` 列表里。文件名：

- `instrdialog/citb_cl_dialogue_tasks_train50_eval10_multiref.json`
- `instrdialogpp/citb_cl_38_random_tasks_train50_eval10_multiref.json`

把它们放在仓库的 `benchmark/instrdialog` 与 `benchmark/instrdialogpp` 下，或者设 `OPD_BENCHMARK_ROOT` 为包含这两个子目录的路径。`benchmarks/instrdialog.py` 与 `benchmarks/instrdialogpp.py` 通过 `core.data.load_continual_stream` 读取。
