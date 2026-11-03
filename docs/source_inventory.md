# 源码清单

训练循环仍是 `core/train.py`，命令入口是 `core/trainer.py`。`scripts/train.py` 只转调它。

| 路径 | 作用 |
|---|---|
| `core/replay_budget.py` | v3c 的 mixed-batch 条数、最大余数、显式余弦 |
| `methods/pgdr.py` | 旧 `quota`、train-holdout 切分和遗忘定义 |
| `methods/opsd.py` | JSD/KL 与 `CE + λ OPSD`。教师描述为 `disable_adapter` |
| `methods/legacy_compat_replay.py` | 调用旧配额和 `CeOpsdReplayMethod` 的 legacy 分支 |
| `methods/selectors.py` | 调用新分配器和同一个 replay 类的 R/U/P |
| `methods/pgdr_opsd.py` | 在选择器上设置 `lambda_opsd`，不新写损失 |
| `baselines/basic_baselines/ce_opsd_replay/method.py` | 真正抽回放的类 |
| `benchmarks/instrdialog.py`、`benchmarks/instrdialogpp.py` | `load_benchmark` / `get_stream` / `validate_dataset` / `get_metadata` |
| `baselines/wrappers/` | 已有 baseline 类的薄重新导出 |
| `scripts/run_v3c_queue.py` | 已结束的 36 run 队列。缺修正协议字段就拒绝启动 |
| `scripts/write_v3c_results.py` | 只从 `final.eval` 写结果 |
| `results/figures/plot_v3c_corrected.py` | 六张图。没有实测就不写文件 |

`core.train._build_baseline_method` 在 `lcia` 上导入 `baselines.advanced_baselines.lcia`。仓库里的类在 `baselines/lcia/method.py`，那条导入路径对不上。这次没有改训练分发，所以没有把 LCIA 包装成已经接好的入口。

已注册、这次没有再包一层的类还有 `ssr`、`onedit`、`citb_init`、`citb_multi`、`citb_ft_no_instr`、`citb_adaptercl`，以及 `ce_opsd_replay` 本身。它们本来就有实现。
