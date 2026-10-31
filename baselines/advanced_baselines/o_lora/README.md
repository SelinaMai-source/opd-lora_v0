# O-LoRA: Orthogonal Subspace Learning for Language Model Continual Learning

状态：downloaded + local smoke-tested with orthogonal hooks。
更新日期：2026-06-15。

## 论文信息

- 论文：Wang et al., Findings of EMNLP 2023 / arXiv:2310.14152
- 链接：https://aclanthology.org/2023.findings-emnlp.715/
- 引用：`@article{wang2023orthogonal, title={Orthogonal Subspace Learning for Language Model Continual Learning}, author={Wang et al.}, journal={arXiv preprint arXiv:2310.14152}, year={2023}}`

## 代码来源与下载状态

- 仓库：https://github.com/cmnfriend/O-LoRA.git
- Commit：`a712f54`
- 许可证/引用信息：源码已落地，许可证仍需在正式复现前复核。
- 下载记录：已下载到 `external_baselines/o_lora`；官方入口轻量检查见 `external_baselines/official_smoke_plan.md`。

## Method 原理

为每个新任务学习新的 LoRA 低秩子空间，并约束新旧 LoRA 子空间正交，以减少无 replay 场景下的遗忘。

## 论文实验设置

官方 README/论文使用 T5-large，并在语言模型持续学习 benchmark 上顺序训练；仓库说明也提到 T5-large，搜索摘要显示支持 T5/LLaMA2 脚本。

## 依赖

官方依赖未能读取；预计需要 transformers/peft/torch、T5-large 或 LLaMA2 权重、原论文任务数据。

## 复现命令

- 本项目 smoke：通过：`python core/train.py --config configs/baselines/smoke_o_lora.yaml`；2 个 mock segment，结果写入 `results/logs/smoke_o_lora.log` 和 `results/tables/smoke_o_lora_segment_metrics.csv`。
- 官方轻量 smoke：`engine.py` 与 `src/run_uie_lora.py` 可 `py_compile`；`--help` 需要独立安装官方依赖。
- 论文或完整复跑：当前机器不能直接复跑论文结果：仍需要原论文 T5/LLaMA2 权重和任务数据；本项目 full scaffold 命令为 `python core/train.py --config configs/baselines/o_lora.yaml`。

## 与本项目接口对接方案

已注册 `baseline_name=o_lora`；本项目创建 segment-wise adapter、冻结旧 adapter，并已补最小 orthogonal regularization/projection hook。当前 hook 以 adapter 向量作为旧子空间代理，正式论文级复现仍需对齐官方 LoRA A/loranew_A 矩阵约束和任务脚本。

## 当前状态

downloaded + local smoke-tested with orthogonal hooks
