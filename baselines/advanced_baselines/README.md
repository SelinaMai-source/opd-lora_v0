# Advanced Baselines

本目录是 advanced baseline 的复现准备区，更新日期：2026-06-14。

这些子目录当前均为 scaffold/待实现，不代表已经完成论文复现。`o_lora`、`progressive_prompts`、`continual_t0` 已有最小 `method.py` 与配置，可做 import/smoke-test 级接入；其余方法仍只有文档 scaffold。统一入口形式为：

```bash
python core/train.py --config configs/<method_config>.yaml
```

当前项目接口约束：

- 数据：`data/processed/*.json`，字段为 `instruction/input/output`，按 `stream[].segment_id` 顺序持续学习。
- 训练：`core/train.py` 只接受 `mode: debug | baseline | ours`，advanced baseline 建议先作为 `mode: baseline` 的新 `baseline_name` 分支接入。
- LoRA：优先复用 `core.models.lora_wrapper` 与 `core.methods.lora_bank`，避免另开训练系统。
- 评估：复用 `core.evaluate`，新增 classification 与 NLG metrics 时应写入同一 per-segment CSV/JSON。

## 方法清单

- `o_lora`: O-LoRA, Orthogonal Subspace Learning for Language Model Continual Learning. 当前为 adapter-per-segment scaffold，正交约束待实现。
- `lb_cl`: LB-CL, Learn More but Bother Less Continual Learning.
- `progressive_prompts`: Progressive Prompts for language model continual learning. 当前为统一训练入口 scaffold，soft prompt 参数模块待实现。
- `lfpt5`: Lifelong Few-shot Language Learning with Prompt Tuning of T5.
- `continual_t0`: Continual-T0 style instruction rehearsal. 当前为小比例 replay scaffold，原 T0/T5 checkpoint 与 mixture 待实现。
- `lamol`: LAMOL generative replay for lifelong language learning.
- `trace_rcl`: TRACE/RCL benchmark and reasoning-augmented continual learning.
- `inf_lora`: InfLoRA-style interference-free low-rank adaptation.
- `adaptercl_dialogue`: Adapter-based continual learning for task-oriented dialogue systems.
- `arper_dialog_nlg`: ARPER for task-oriented dialogue NLG.
- `tm_bnnm_dialog_nlg`: Text-Mixup + BNNM for continual dialog generation.

## 优先级建议

1. O-LoRA、LB-CL、Progressive Prompts：最适合与本项目 LoRA Bank/router/anti-overlap 直接比较。
2. LFPT5、Continual-T0、LAMOL、TRACE/RCL：强 instruction/replay/benchmark 对照。
3. InfLoRA、AdapterCL、ARPER、TM-BNNM：作为 PEFT 或 dialogue/NLG 迁移型补充基线。
