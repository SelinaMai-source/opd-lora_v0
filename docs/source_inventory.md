# 源码清单

没有新建空的 `core/memory.py` 或 `core/config.py`。记忆和配置仍在原来的文件里。

| 路径 | 作用 |
|---|---|
| `core/trainer.py` | 命令入口 `python -m core.trainer` |
| `core/train.py` | 训练循环，按 `baseline_name` 调用现有类 |
| `core/data.py` | `Example`、`Segment`、流加载 |
| `core/replay_budget.py` | v3c mixed-batch 条数、最大余数、显式余弦 |
| `methods/pgdr.py` | 旧 `quota`、train-holdout 切分和遗忘定义 |
| `methods/opsd.py` | JSD/KL 与 `CE + λ OPSD` |
| `methods/legacy_compat_replay.py` | 旧配额和 `uniform` / `proto` |
| `methods/selectors.py` | R / U / P，调用同一个 replay 类 |
| `methods/pgdr_opsd.py` | 在选择器上设置 `lambda_opsd` |
| `baselines/basic_baselines/ce_opsd_replay/method.py` | 真正抽回放的类 |
| `baselines/sequential.py` 等 14 个公开模块 | 只调用现有类。同名包通过 `__path__` 继续可导入 |
| `baselines/wrappers/` | 转调上面的公开模块。实现目录不删 |
| `benchmarks/instrdialog.py`、`benchmarks/instrdialogpp.py` | `load_benchmark` / `get_stream` / `validate_dataset` / `get_metadata` |
| `configs/experiments/corrected_factorial.yaml` | 已完成的 36 个 run。数据根和结果根是 `${DATA_ROOT}` / `${RESULTS_ROOT}` |
| `configs/v3c_*.yaml` | 已经跑过的单次配置，路径保持原样 |
| `scripts/train.py` | 转调 `core.trainer.main` |
| `scripts/run_v3c_queue.py` | 已结束的队列。缺 `train_holdout` 或 `budget_policy: unified` 就拒绝启动 |
| `scripts/write_v3c_results.py` | 只从 `final.eval` 写结果 |
| `results/figures/plot_v3c_corrected.py` | 六张图 |
| `results/manifest.csv` | 公开清单。路径列是仓库相对路径 |

`core.train._build_baseline_method` 在 `lcia` 上仍导入 `baselines.advanced_baselines.lcia`。仓库里的类在 `baselines/lcia/method.py`。这次没有改训练分发。
