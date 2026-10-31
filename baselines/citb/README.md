# CITB 官方数据 + 本仓库对照算法

本目录不存放 1600+ SuperNI task JSON。官方仓库在：

- `/root/autodl-tmp/CITB/`
- 符号链接：`official` → 上述路径

常用子路径：

- `official/data/tasks/`：原始 task JSON（`output` 常为 `list[str]`）
- `official/data/splits/CIT_splits/cl_dialogue_tasks.txt`：InstrDialog 19 段顺序
- `official/data/splits/CIT_splits/cl_38_random_tasks.txt`：InstrDialog++ 38 段顺序

processed 流写在 `../instrdialog/` 与 `../instrdialogpp/`，由 `scripts/rebuild_benchmark_streams.py` 从这些原始 JSON 重打包。

对照算法（`core.train` 注册名 → 子目录）：

| baseline_name | 目录 | 口径 |
|---|---|---|
| `citb_init` | `init/` | Instruct 只评不训 |
| `citb_multi` | `multi/` | 第 0 段联合 SFT 全流 train |
| `citb_ft_no_instr` | `ft_no_instr/` | Sequential，train/eval 去掉 instruction |
| `citb_replay` | `replay/` | 每段定额 K 条记忆，下一段 current+memory |
| `citb_ewc` | `ewc/` | K 记忆算 Fisher，`reg=0.01` |
| `citb_l2` | `l2/` | 锚定 θ*，`reg=0.01` |
| `citb_agem` | `agem/` | 记忆梯度投影，K=50/10 |
| `citb_adaptercl` | `adaptercl/` | 每段一个 LoRA；eval 按 segment_id 切 adapter |

配置：`configs/v2pre_citb_<method>_instrdialog_seed456.yaml`（++ 同理）。Replay-10 / AGEM-10 / AdapterCL-200 分别写 `replay.k` 或 `lora.r`。FT_INSTR 沿用已有 `sequential_lora` run，不在此重跑。
