# TM-BNNM: Text-Mixup + Batch Nuclear-Norm Maximization for Continual Dialog Generation

状态：scaffold-only / blocked。
更新日期：2026-06-15。

## 论文信息

- 论文：arXiv:2403.10894, 2024
- 链接：https://arxiv.org/abs/2403.10894
- 引用：`@article{wang2024towards, title={Towards Robustness and Diversity: Continual Learning in Dialog Generation with Text-Mixup and Batch Nuclear-Norm Maximization}, journal={arXiv preprint arXiv:2403.10894}, year={2024}}`

## 代码来源与下载状态

- 仓库：未找到官方公开代码仓库
- Commit：无
- 许可证/引用信息：无代码许可证。
- 下载记录：搜索未找到该方法官方仓库；相关搜索结果只出现 ARPER 仓库和其他 TOD CL 仓库，未将其伪造为 TM-BNNM 实现。

## Method 原理

在 replay/current batch 上做 Text-Mixup 降低 memory overfit，并最大化 batch hidden-state nuclear norm 提升表示多样性、缓解模式坍缩。

## 论文实验设置

37-domain task-oriented dialog dataset 和 10-domain DailyDialog continual generation；报告生成质量和遗忘。

## 依赖

无官方依赖；本项目复现需 torch/transformers、hidden state access、SVD/nuclear norm loss、TOD37/DailyDialog 数据。

## 复现命令

- 本项目 smoke / scaffold：未运行；无官方代码，本项目未实现 method.py。
- 论文或完整复跑：当前机器不能直接复跑论文结果：无官方代码、数据和实现缺失；需基于论文重写 Text-Mixup/BNNM loss。

## 与本项目接口对接方案

建议先实现 replay batch hidden-state API，再添加 text mixup collator 和 -lambda*||Z||_* loss。

## 当前状态

scaffold-only / blocked
