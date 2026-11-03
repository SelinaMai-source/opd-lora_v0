# opd-lora v3

修正回放配额和真正余弦检索之后的研究代码。v3c 的 36 个 run 已经有 `final_metrics.json`。R、U、P 是同一条 mixed-batch ρ=0.20 上的三种选择器。它们和旧的 9 对 8（`uniform` / `proto`，含 `v3h_*`）以及 Protocol A 不能放进同一张均值表。

## 安装

在已经包含 torch、transformers、peft 的环境里，从仓库根目录：

```bash
pip install -e . --no-deps
```

不要升级这三样。`requirements.lock` 是当前 Linux + CUDA 12.8 环境里已经装好的版本记录，不是通用 lock，也不能用来重装出另一台机器的 CUDA 轮子。

本仓库没有另发 LICENSE，也不把第三方代码或 CITB 数据重新许可。没有 DOI。

## 外部依赖

下面两样不在 git 里，CPU 测试也不读取它们。

- 数据：把两个 multiref json 放在 `benchmark/instrdialog` 与 `benchmark/instrdialogpp`，或设置 `OPD_BENCHMARK_ROOT`。因子描述里的数据根是环境变量 `DATA_ROOT`。来源见 `data/README.md`。
- Llama 3.1 8B Instruct 权重：配置里的 `assets/pretrained/Meta-Llama-3.1-8B-Instruct` 需要自行获取。

已完成 run 的 checkpoint、预测和 `final_metrics.json` 留在数据盘，不在本仓库。v3c 的 36 个完成目录在 `/root/autodl-tmp/rp_lora_v3_runs/runs/`。更早的 run 在同一结果根以及 `/root/autodl-tmp/rp_lora_v0/results/runs/`。公开清单 `results/manifest.csv` 只用仓库相对路径。

## 训练入口

下面两条调用同一个 `core.trainer.main`：

```bash
python -m core.trainer --config configs/v3c_R_ce_instrdialog_seed456.yaml
python scripts/train.py --config configs/v3c_R_ce_instrdialog_seed456.yaml
```

36 个已完成 run 的矩阵在 `configs/experiments/corrected_factorial.yaml`。单次训练仍用 `configs/v3c_*.yaml`。那些文件里的路径没有改写。结果根在因子文件里写成 `${RESULTS_ROOT}`；没有展开时读同名环境变量。

`scripts/run_v3c_queue.py` 是已经跑完的队列。配置缺 `monitor: train_holdout` 或 `budget_policy: unified` 时它拒绝启动。不要为了覆盖 `v3h_*` 或已有 `final_metrics.json` 重开训练。

## CPU 测试

不加载 8B，也不读私人数据盘：

```bash
python -m unittest tests.test_replay_budget tests.test_prototype_retrieval tests.test_public_wrappers tests.test_benchmark_isolation tests.test_corrected_factorial
python scripts/smoke_v3c_cpu.py
```

`.github/workflows/tests.yml` 跑同一组 unittest。它不下载权重。

## 结果

`docs/ACL_v3_Corrected_Method_Results.md` 是 36 行实测。图在 `results/figures/`，重画脚本是 `results/figures/plot_v3c_corrected.py`。要从原始 json 重写表时，把 `OPD_RUNS_DIR` 指到含 `v3c_*/final_metrics.json` 的目录，再运行 `python scripts/write_v3c_results.py`。这两步依赖数据盘上的实测文件。

方法、配额和旧协议见 `docs/method.md`、`docs/algorithms.md`、`docs/legacy_vs_corrected.md`。源码对应关系见 `docs/source_inventory.md`，实现边界见 `docs/repository_audit.md`。
