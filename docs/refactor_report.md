# 整理记录

选择器和配额的行为没有改。`core/replay_budget.py` 与 `baselines/basic_baselines/ce_opsd_replay/method.py` 的抽样没有在这次提交里重写。没有新开训练，没有覆盖 `v3h_*` 或 `final_metrics.json`。

## 已完成的 36 个 run 之前

- `benchmarks/` 复用 `core.data.load_continual_stream`。
- `methods/legacy_compat_replay.py`、`methods/selectors.py`、`methods/pgdr_opsd.py` 调用现有类。
- `scripts/train.py` 转调 `core.trainer.main`。
- `pyproject.toml` 让 `pip install -e . --no-deps` 能导入这些包，不升级 torch、transformers、peft。
- 36 行实测和图已经在 `docs/ACL_v3_Corrected_Method_Results.md` 与 `results/figures/`。

## 这次只搬家和加入口

- 14 个薄封装放到 `baselines/sequential.py`、`citb_replay.py`、`ewc.py`、`agem.py`、`l2.py`、`inscl.py`、`kpig.py`、`clora.py`、`gorp.py`、`sdft.py`、`gainlora.py`、`treelora.py`、`sapt.py`、`saber.py`。
- 同名实现目录保留。公开模块把 `__path__` 指到该目录，所以 `baselines.inscl.method` 仍是原来的类。
- `baselines/wrappers/` 保留，改为转调这些公开模块。
- `configs/experiments/corrected_factorial.yaml` 只描述这 36 个已完成 run。数据根和结果根写 `${DATA_ROOT}` / `${RESULTS_ROOT}`。单次 yaml 的绝对路径没有改。
- `results/manifest.csv` 去掉服务器绝对路径。仓库里没有的权重、预测和 metrics 文件留空。
- `.gitignore` 排除 `.venv`、`__pycache__`、`.safetensors`、`.pt`、`.env`。
- `requirements.lock` 只记录当前 Linux + CUDA 12.8 环境里已经装好的版本。
- `.github/workflows/tests.yml` 只跑不加载 8B、不读私人数据盘的 CPU 测试。
- 没有新建 `core/memory.py` 或 `core/config.py`。没有新增 LICENSE，也没有写 DOI。

没做：

- 不为 LAMOL、TRACE/RCL、InfLoRA、LFPT5、ARPER、TM-BNNM 建训练类。
- 不把 O-LoRA、LB-CL、Progressive Prompts、Continual-T0 的 scaffold 包装成论文实现。
- 不给 `bank_no_router` 补算法。训练循环里它返回 `object()`。
