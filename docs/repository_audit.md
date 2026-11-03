# 仓库审计

这次只把已经能训练的类挂到公开路径，并补上可安装入口。选择器、配额和 `v3c_*` 的 `final_metrics.json` 没有改。

## 已实现

训练入口是 `python -m core.trainer`。`scripts/train.py` 转调同一个 `main`。

v3c 的 R / U / P 在 `methods/selectors.py`，调用 `core/replay_budget.py` 和 `baselines/basic_baselines/ce_opsd_replay/method.py`。旧 `uniform` / `proto` 在 `methods/legacy_compat_replay.py`，仍走 `methods/pgdr.py` 的 9 对 8。

下面这些目录里有训练类，这次没有再抄一份循环：

- `baselines/basic_baselines/` 下的 sequential LoRA、replay、OPSD、CE+OPSD、metric、gold、neg、seg
- `baselines/citb/` 下的 init、multi、ft_no_instr、replay、ewc、agem、l2、adaptercl
- `baselines/lcia/method.py`
- `baselines/inscl`、`kpig`、`clora`、`gorp`、`sdft`、`gainlora`、`treelora`、`sapt`、`saber`、`ssr`、`onedit`

## 已归位

这 14 个文件只调用上一节的类：

| 公开模块 | 现有类 |
|---|---|
| `baselines/sequential.py` | `SequentialLoRAMethod` |
| `baselines/citb_replay.py` | `CITBReplayMethod` |
| `baselines/ewc.py` | `CITBEWCMethod` |
| `baselines/agem.py` | `CITBAGEMMethod` |
| `baselines/l2.py` | `CITBL2Method` |
| `baselines/inscl.py` | `InsCLMethod` |
| `baselines/kpig.py` | `KPIGMethod` |
| `baselines/clora.py` | `CLoRAMethod` |
| `baselines/gorp.py` | `GORPMethod` |
| `baselines/sdft.py` | `SDFTMethod` |
| `baselines/gainlora.py` | `GainLoRAMethod` |
| `baselines/treelora.py` | `TreeLoRAMethod` |
| `baselines/sapt.py` | `SAPTMethod` |
| `baselines/saber.py` | `SABERMethod` |

同名目录还在。公开模块设置 `__path__`，`from baselines.inscl.method import InsCLMethod` 仍然读到原来的类。`baselines/wrappers/` 只转调这些公开模块。

`benchmarks/instrdialog.py` 与 `benchmarks/instrdialogpp.py` 复用 `core.data`。`configs/experiments/corrected_factorial.yaml` 只描述已经跑完的 36 个 v3c run。单次训练仍用 `configs/v3c_*.yaml`，那些文件里的路径没有改。

`results/manifest.csv` 的路径列只保留仓库里的文件。权重、预测和 `final_metrics.json` 不在 git 里，对应格子是空的。36 行数字在 `docs/ACL_v3_Corrected_Method_Results.md`。

## 未实现

没有忠实训练实现，所以没有新建空类：

- LAMOL
- TRACE / RCL
- InfLoRA
- LFPT5
- ARPER
- TM-BNNM
- `advanced_baselines/adaptercl_dialogue`（CITB 的 `citb_adaptercl` 是另一份实现，不在此列）
- O-LoRA、LB-CL、Progressive Prompts、Continual-T0：类写明是 scaffold
- `bank_no_router`：`core/train.py` 返回 `object()`

SSR、OnEDIT、LCIA 以及 CITB init / multi / ft_no_instr / adaptercl 已有类，不在上面的 14 个公开文件名里，也没有另做一层空包装。

## Legacy

`uniform` 在 45 条 current 上是 9 条回放，`proto` 是 4+4=8。v3c 的 R、U、P 共用 mixed-batch ρ=0.20，同一位置是 11 条。`v3h_*` 和 Protocol A 不进 v3c 的均值。队列在缺少 `monitor: train_holdout` 或 `budget_policy: unified` 时拒绝启动。

checkpoint 和预测留在数据盘，不在本仓库：

- v3c 的 36 个完成目录：`/root/autodl-tmp/rp_lora_v3_runs/runs/v3c_*`
- 更早的 v3 / v3h 与清单里的其他 run：`/root/autodl-tmp/rp_lora_v3_runs/runs/` 与 `/root/autodl-tmp/rp_lora_v0/results/runs/`
