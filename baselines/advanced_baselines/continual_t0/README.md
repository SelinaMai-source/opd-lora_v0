# Continual-T0: Progressively Instructing 50+ Tasks to Language Models

状态：blocked + local smoke-tested scaffold。
更新日期：2026-06-15。

## 论文信息

- 论文：Scialom et al., EMNLP 2022 / arXiv:2205.12393
- 链接：https://aclanthology.org/2022.emnlp-main.410/
- 引用：`@inproceedings{scialom2022fine, title={Fine-tuned Language Models are Continual Learners}, author={Scialom et al.}, booktitle={EMNLP}, year={2022}}`

## 代码来源与下载状态

- 仓库：https://github.com/ThomasScialom/T0_continual_learning.git
- Commit：无有效本地 commit；clone 失败后留下 partial 目录
- 许可证/引用信息：未能下载 LICENSE，待网络恢复后复核。
- 下载记录：git clone 失败：GitHub 443 timeout；git ls-remote exit=124。

## Method 原理

从 instruction-tuned T0/T5 出发，顺序学习 8 个新生成任务，并用约 1% rehearsal buffer 保持旧任务和 zero-shot 能力。

## 论文实验设置

T0 原始 50 任务 + 12 zero-shot 评估 + 8 个 NLG 新任务；论文报告 CT0 在 70 数据集范围内保持能力。

## 依赖

T0/T5 checkpoint、论文任务集合、transformers/datasets、rehearsal buffer 构造。

## 复现命令

- 本项目 smoke / scaffold：通过：python core/train.py --config configs/baselines/smoke_continual_t0.yaml；当前仅实现 1% instruction replay wiring，不复现原 T0/T5。
- 论文或完整复跑：当前机器不能直接复跑论文结果：官方代码未下载，T0 checkpoint 和 50+8 任务数据未准备；本项目 full scaffold 命令为 python core/train.py --config configs/baselines/continual_t0.yaml。

## 与本项目接口对接方案

已注册 baseline_name=continual_t0；需补 T0/T5 任务混合、rehearsal 采样和原论文评估。

## 当前状态

blocked + local smoke-tested scaffold
