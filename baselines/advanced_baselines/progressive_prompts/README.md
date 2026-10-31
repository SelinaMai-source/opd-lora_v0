# Progressive Prompts: Continual Learning for Language Models

状态：blocked + local smoke-tested scaffold。
更新日期：2026-06-15。

## 论文信息

- 论文：Razdaibiedina et al., ICLR 2023 / arXiv:2301.12314
- 链接：https://openreview.net/forum?id=UJTgQBc91_
- 引用：`@inproceedings{razdaibiedina2023progressive, title={Progressive Prompts: Continual Learning for Language Models}, author={Razdaibiedina et al.}, booktitle={ICLR}, year={2023}}`

## 代码来源与下载状态

- 仓库：https://github.com/arazd/ProgressivePrompts.git
- Commit：无有效本地 commit；clone 未落地
- 许可证/引用信息：WebSearch 摘要显示 Apache-2.0；本地未下载 LICENSE。
- 下载记录：git clone --depth 1 失败：GitHub 443 timeout；git ls-remote exit=124。

## Method 原理

冻结 backbone，为每个任务学习新的 soft prompt，并把历史 prompt 与当前 prompt 级联，降低遗忘且不保存 replay 数据。

## 论文实验设置

官方仓库包含 BERT/T5 codebase；论文在标准 CL 分类/NLP 任务上报告 T5 平均准确率提升。

## 依赖

官方依赖未能读取；预计需要 torch/transformers、T5/BERT 权重和论文数据。

## 复现命令

- 本项目 smoke / scaffold：通过：python core/train.py --config configs/baselines/smoke_progressive_prompts.yaml；当前只是 prompt schedule 占位，不训练真实 soft prompt。
- 论文或完整复跑：当前机器不能直接复跑论文结果：官方代码未下载；本项目 full scaffold 命令为 python core/train.py --config configs/baselines/progressive_prompts.yaml，但 soft prompt module/级联输入尚未实现。

## 与本项目接口对接方案

已注册 baseline_name=progressive_prompts；需新增 prompt embedding pool、冻结旧 prompt、oracle/router prompt selection。

## 当前状态

blocked + local smoke-tested scaffold
