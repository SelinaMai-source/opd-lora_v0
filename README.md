# opd-lora v3

修正回放配额和真正余弦检索之后的研究代码。v3c 的 36 个 run 已经有 `final_metrics.json`。旧的 `v3h_*` 与 Protocol A 分开保留，不混进 v3c 的均值。

## 安装

在已经包含 torch、transformers、peft 的环境里，从仓库根目录：

```bash
pip install -e . --no-deps
```

不要升级这三样。不需要把作者机器上的其他目录放进 `PYTHONPATH`。

数据不在 git 里。把两个 multiref json 放在 `benchmark/instrdialog` 与 `benchmark/instrdialogpp`，或设置 `OPD_BENCHMARK_ROOT`。来源和许可边界见 `data/README.md`。

## 两条入口

下面两条调用同一个 `core.trainer.main`，都要 `--config`：

```bash
python -m core.trainer --config configs/<run>.yaml
python scripts/train.py --config configs/<run>.yaml
```

不加载 8B 的检查：

```bash
python scripts/smoke_v3c_cpu.py
python -m unittest tests.test_replay_budget tests.test_prototype_retrieval tests.test_holdout_replay_opsd tests.test_public_wrappers
```

## 结果

`docs/ACL_v3_Corrected_Method_Results.md` 是 36 行实测。图在 `results/figures/`，脚本是旁边的 `plot_v3c_corrected.py`。重画时把 `OPD_RUNS_DIR` 指到含 `v3c_*/final_metrics.json` 的目录。

方法、配额和旧协议的差别见 `docs/method.md`、`docs/algorithms.md`、`docs/legacy_vs_corrected.md`。

本仓库不编造 DOI，也不用一份新的 LICENSE 重新许可外部代码或 CITB 数据。
