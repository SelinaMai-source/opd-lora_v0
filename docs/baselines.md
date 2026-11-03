# Baseline 包装

`baselines/wrappers/` 里的每个文件只导入仓库里已经存在的类，并提供 `build(cfg)`。训练仍由 `core.train._build_baseline_method` 分发到原来的类。

## 已包装

| baseline_name | 类 |
|---|---|
| `sequential_lora` | `SequentialLoRAMethod` |
| `citb_replay` | `CITBReplayMethod` |
| `citb_ewc` | `CITBEWCMethod` |
| `citb_agem` | `CITBAGEMMethod` |
| `citb_l2` | `CITBL2Method` |
| `inscl` | `InsCLMethod` |
| `kpig` | `KPIGMethod` |
| `clora` | `CLoRAMethod` |
| `gorp` | `GORPMethod` |
| `sdft` | `SDFTMethod` |
| `gainlora` | `GainLoRAMethod` |
| `treelora` | `TreeLoRAMethod` |
| `sapt` | `SAPTMethod` |
| `saber` | `SABERMethod` |

仓库里还有类、这次不另做一层包装的：`ssr`、`onedit`、`lcia`（文件在 `baselines/lcia/method.py`）、`citb_init`、`citb_multi`、`citb_ft_no_instr`、`citb_adaptercl`，以及 OPSD / replay 家族。

## Not Implemented

下面这些没有忠实的训练实现。这次没有为它们新建 `method.py`。

- LAMOL：只有 `baselines/advanced_baselines/lamol/README.md` 和外部仓记录。
- TRACE / RCL：只有 `baselines/advanced_baselines/trace_rcl/README.md` 和外部仓记录。
- O-LoRA、LB-CL、Progressive Prompts、Continual-T0：类写明是 scaffold，不包装成论文实现。
- `bank_no_router`：`core/train.py` 直接 `return object()`，没有独立算法。
