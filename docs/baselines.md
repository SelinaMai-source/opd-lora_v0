# Baseline 包装

`baselines/sequential.py`、`citb_replay.py`、`ewc.py`、`agem.py`、`l2.py`、`inscl.py`、`kpig.py`、`clora.py`、`gorp.py`、`sdft.py`、`gainlora.py`、`treelora.py`、`sapt.py`、`saber.py` 只导入仓库里已经存在的类，并提供 `build(cfg)`。训练仍由 `core.train._build_baseline_method` 分发到原来的类。`baselines/wrappers/` 转调这些文件。同名实现目录不删。

## 已包装

| 公开文件 | baseline_name | 类 |
|---|---|---|
| `baselines/sequential.py` | `sequential_lora` | `SequentialLoRAMethod` |
| `baselines/citb_replay.py` | `citb_replay` | `CITBReplayMethod` |
| `baselines/ewc.py` | `citb_ewc` | `CITBEWCMethod` |
| `baselines/agem.py` | `citb_agem` | `CITBAGEMMethod` |
| `baselines/l2.py` | `citb_l2` | `CITBL2Method` |
| `baselines/inscl.py` | `inscl` | `InsCLMethod` |
| `baselines/kpig.py` | `kpig` | `KPIGMethod` |
| `baselines/clora.py` | `clora` | `CLoRAMethod` |
| `baselines/gorp.py` | `gorp` | `GORPMethod` |
| `baselines/sdft.py` | `sdft` | `SDFTMethod` |
| `baselines/gainlora.py` | `gainlora` | `GainLoRAMethod` |
| `baselines/treelora.py` | `treelora` | `TreeLoRAMethod` |
| `baselines/sapt.py` | `sapt` | `SAPTMethod` |
| `baselines/saber.py` | `saber` | `SABERMethod` |

仓库里还有类、这次不另做一层公开文件的：`ssr`、`onedit`、`lcia`、`citb_init`、`citb_multi`、`citb_ft_no_instr`、`citb_adaptercl`，以及 OPSD / replay 家族。

## 未实现

下面这些没有忠实的训练实现。没有为它们新建类。

- LAMOL
- TRACE / RCL
- InfLoRA
- LFPT5
- ARPER
- TM-BNNM
- `advanced_baselines/adaptercl_dialogue`
- O-LoRA、LB-CL、Progressive Prompts、Continual-T0：scaffold
- `bank_no_router`：`core/train.py` 返回 `object()`
