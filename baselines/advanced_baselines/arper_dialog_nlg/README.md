# ARPER: Continual Learning for NLG in Task-oriented Dialog Systems

状态：blocked / partial checkout。
更新日期：2026-06-15。

## 论文信息

- 论文：Mi et al., Findings of EMNLP 2020 / arXiv:2010.00910
- 链接：https://aclanthology.org/2020.findings-emnlp.310/
- 引用：`@inproceedings{mi2020continual, title={Continual Learning for Natural Language Generation in Task-oriented Dialog Systems}, author={Mi et al.}, booktitle={Findings of EMNLP}, year={2020}}`

## 代码来源与下载状态

- 仓库：https://github.com/MiFei/Continual-Learning-for-NLG.git
- Commit：远程 master 可见 99019defe6bf35e8459ca6abd6f25882724bc956；本地 fallback partial，无 HEAD
- 许可证/引用信息：未能完整下载 LICENSE，待网络恢复后复核。
- 下载记录：git ls-remote 可达；clone 过程被网络阻塞并留下 baselines/advanced_baselines/arper_dialog_nlg/external partial NO_HEAD。

## Method 原理

Adaptively Regularized Prioritized Exemplar Replay：优先 exemplar replay 加自适应 EWC/Fisher 正则，用于 MultiWOZ task-oriented NLG。

## 论文实验设置

MultiWOZ-2.0 NLG continual learning，按 domain 或 dialogue-act split，报告 BLEU/ERR 等 NLG 指标。

## 依赖

官方依赖未能读取；预计 PyTorch、MultiWOZ preprocess、NLG metrics。

## 复现命令

- 本项目 smoke / scaffold：未运行；无有效 checkout。
- 论文或完整复跑：当前机器不能直接复跑论文结果：代码未完整下载，MultiWOZ 数据/预处理和依赖未准备。

## 与本项目接口对接方案

本项目可实现 exemplar scoring、prioritized replay、Fisher/EWC，并将 dialogue-act-to-text 转为 instruction triples。

## 当前状态

blocked / partial checkout
