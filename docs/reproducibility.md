# 可复现

## 环境

使用已经装好 torch、transformers、peft 的环境。不要为了本仓库升级这三样。`pyproject.toml` 不声明它们的新版本。

仓库根目录：

```bash
pip install -e . --no-deps
python -m unittest tests.test_replay_budget tests.test_prototype_retrieval tests.test_public_wrappers
python scripts/smoke_v3c_cpu.py
```

`scripts/smoke_v3c_cpu.py` 不加载 8B。它检查 45→11、6+5，以及真余弦会先取高分的段、compat_uniform 不会被说成同一行为。

两条训练入口是同一个 `main`：

```bash
python -m core.trainer --config configs/<run>.yaml
python scripts/train.py --config configs/<run>.yaml
```

脚本会把仓库根放进 `sys.path`。不需要把作者机器上的其他目录写进 `PYTHONPATH`。

## 数据与 run

数据不在 git 里。把 multiref json 放在 `benchmark/instrdialog` 与 `benchmark/instrdialogpp`，或设置 `OPD_BENCHMARK_ROOT`。说明见 `data/README.md`。

已完成的 v3c 指标在各自 run 的 `final_metrics.json` 的 `final.eval.*`。重画图表时设置 `OPD_RUNS_DIR` 指向含这些 run 目录的路径，然后：

```bash
python scripts/write_v3c_results.py
python results/figures/plot_v3c_corrected.py
```

正式 36 个 run 已经跑完。不要重跑来覆盖 `v3h_*` 或已有 `final_metrics.json`。

## 种子

新实验默认 seed 456。本修正协议的稳健性只用 789 和 42，而且只用于 v3c 的 R/U/P。`holdout_seed` 固定 456，不随训练 seed 改变。

## 指标

11 列只从 `final.eval` 读取，保留三位小数。Seen-Avg Acc 越高越好。Forgetting 越低越好，下降才是改善。多 gold 取最佳对，格式归一打开。细节在 `docs/method.md` 与结果文件。
