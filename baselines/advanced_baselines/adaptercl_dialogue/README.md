# AdapterCL / Residual Adapter for Task-Oriented Dialogue Continual Learning

状态：blocked。
更新日期：2026-06-15。

## 论文信息

- 论文：Madotto et al., EMNLP 2021
- 链接：https://aclanthology.org/2021.emnlp-main.590/
- 引用：`@inproceedings{madotto2021continual, title={Continual Learning in Task-Oriented Dialogue Systems}, author={Madotto et al.}, booktitle={EMNLP}, year={2021}}`

## 代码来源与下载状态

- 仓库：https://github.com/andreamad8/ToDCL.git
- Commit：远程 main 可见 e70c1edf937f6eb570296ea2897dbc8d6815bc6d；本地 clone 未成功
- 许可证/引用信息：未能下载 LICENSE，待网络恢复后复核。
- 下载记录：git ls-remote 可达；clone 到 external_baselines/adaptercl_dialogue 和 fallback external 均 GitHub 443 timeout。

## Method 原理

为 TOD domain/task 学习 residual adapters，并用 entropy classifier 选择测试时 adapter；同时比较 replay、EWC、AGEM、LAMOL 等。

## 论文实验设置

37-domain TOD benchmark，覆盖 INTENT/DST/NLG/E2E；示例 python train.py --task_type NLG --CL ADAPTER --bottleneck_size 50 ...。

## 依赖

官方依赖未下载；预计 torch/transformers/GPT-2/MultiWOZ/TOD37 数据。

## 复现命令

- 本项目 smoke / scaffold：未运行；无有效 checkout。
- 论文或完整复跑：当前机器不能直接复跑论文结果：代码未下载，TOD37/MultiWOZ 数据和依赖未准备。

## 与本项目接口对接方案

可用本项目 per-segment LoRA adapter 近似 residual adapter，并增加 oracle adapter / router-selected adapter 两种评估。

## 当前状态

blocked
