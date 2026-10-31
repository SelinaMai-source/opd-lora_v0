# TRACE / RCL: Reasoning-Augmented Continual Learning

状态：downloaded + smoke partially blocked。
更新日期：2026-06-15。

## 论文信息

- 论文：Wang et al., arXiv/OpenReview 2023 / arXiv:2310.06762
- 链接：https://openreview.net/forum?id=xelrLobW0n
- 引用：`@article{wang2023trace, title={TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models}, author={Wang et al.}, journal={arXiv preprint arXiv:2310.06762}, year={2023}}`

## 代码来源与下载状态

- 仓库：https://github.com/BeyonderXX/TRACE.git
- Commit：462e39f616134f4f819efeb3baea8638c03c7db4
- 许可证/引用信息：Apache License 2.0（本地 LICENSE 已读取）。
- 下载记录：成功下载到 external_baselines/trace_rcl，commit 462e39f616134f4f819efeb3baea8638c03c7db4。

## Method 原理

TRACE 是 8 任务 LLM CL benchmark；RCL 用任务分析/推理路径增强训练样本，保护通用能力、指令跟随和安全能力。

## 论文实验设置

官方 README 要求 CUDA 12.2、torch 2.0.1、deepspeed；脚本默认 LLaMA-2-7B-chat、8 GPU、TRACE 数据路径和 OpenCompass/GPT-4 评估。

## 依赖

datasets, sentencepiece, accelerate, deepspeed, transformers==4.31.0, CUDA 11/12 包、peft、qpth、quadprog 等。

## 复现命令

- 本项目 smoke / scaffold：training/main.py py_compile 通过；python training/main.py --help 失败：ModuleNotFoundError: No module named deepspeed。未安装重依赖，未启动训练。
- 论文或完整复跑：当前机器不能直接复跑论文结果：缺 TRACE 数据、LLaMA-2-7B-chat 权重、deepspeed、脚本写死 8 卡路径；官方示例 bash scripts/train_seq_cl.sh / bash scripts/infer_seq.sh 需先改 data_path/model_path/output_dir。

## 与本项目接口对接方案

可先写 TRACE -> data/processed converter，再用 baseline_name=trace_rcl 或 LoRA/replay 分支复现 RCL rationale augmentation。

## 当前状态

downloaded + smoke partially blocked
