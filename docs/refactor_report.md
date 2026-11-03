# 整理记录

队列结束之后才加这些文件。`core/train.py` 和 `baselines/basic_baselines/ce_opsd_replay/method.py` 的训练行为没有在这一步改动。

做了：

- `benchmarks/` 复用 `core.data.load_continual_stream`。路径来自参数、config 或 `OPD_BENCHMARK_ROOT`。
- `baselines/wrappers/` 重新导出 sequential、CITB replay/ewc/agem/l2、InsCL、KPIG、CLoRA、GORP、SDFT、GainLoRA、TreeLoRA、SAPT、SABER。包装文件不写新的损失或采样。
- 没有把 `baselines/inscl.py` 这类文件放在包目录旁边，否则会挡住已有的 `baselines/inscl/`。
- `methods/legacy_compat_replay.py`、`methods/selectors.py`、`methods/pgdr_opsd.py` 调用现有类和对应的配额函数。
- `scripts/train.py` 转调 `core.trainer.main`。
- `pyproject.toml` 让 `pip install -e . --no-deps` 能导入这些包，不升级 torch。

没做：

- 不为 LAMOL、TRACE/RCL 建训练类。它们只有外部仓说明。
- 不把 O-LoRA、LB-CL、Progressive Prompts、Continual-T0 的 scaffold 包装成忠实实现。
- 不给 `bank_no_router` 补算法文件。训练循环里它返回的是 `object()`。
- 不加覆盖第三方代码的 LICENSE。LAMOL 的说明写 MIT，TRACE 的说明写 Apache-2.0，那些是外部检出的许可证，不是本仓库可以重新授予的。CITB 检出没有 LICENSE 文件。

结果摘要单独写在 `docs/ACL_v3_Corrected_Method_Results.md`，数字只来自 v3c 的 `final_metrics.json`。
