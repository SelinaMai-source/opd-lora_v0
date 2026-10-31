# InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning

状态：blocked / partial checkout。
更新日期：2026-06-15。

## 论文信息

- 论文：Liang and Li, CVPR 2024
- 链接：https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html
- 引用：`@inproceedings{liang2024inflora, title={InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning}, author={Liang, Yan-Shuo and Li, Wu-Jun}, booktitle={CVPR}, year={2024}}`

## 代码来源与下载状态

- 仓库：https://github.com/liangyanshuo/InfLoRA.git
- Commit：远程 main 可见 e08b00edd54f2f10cf2f9826eae7d44fdcb6354b；本地 clone partial，无 HEAD
- 许可证/引用信息：未能完整下载 LICENSE，待网络恢复后复核。
- 下载记录：git ls-remote 可达；clone 到 external_baselines/inf_lora 和 fallback external 均超时/无 HEAD，未获得可用工作树。

## Method 原理

将预训练权重重参数化到低秩子空间，并构造对旧任务近似无干扰、对新任务有适应性的训练子空间。

## 论文实验设置

原论文主要为视觉 continual learning（ImageNet-R/CIFAR100/DomainNet 等），不是 NLP 原生设置。

## 依赖

官方依赖未能读取；预计 torch/timm/vision CL 数据。迁移到 LLM 需 transformers/peft 和 hidden/gradient 子空间统计。

## 复现命令

- 本项目 smoke / scaffold：未运行；无有效 checkout。
- 论文或完整复跑：当前机器不能直接复跑论文结果：代码未完整下载，且论文任务为视觉；LLM 迁移需要重新定义特征协方差和 projection。

## 与本项目接口对接方案

建议作为低优先级迁移型 PEFT baseline，先完成本项目 LoRA 梯度/特征投影 API 后再接入。

## 当前状态

blocked / partial checkout
