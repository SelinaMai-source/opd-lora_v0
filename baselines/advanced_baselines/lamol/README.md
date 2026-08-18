# LAMOL: Language Modeling for Lifelong Language Learning

状态：downloaded + smoke partially blocked。
更新日期：2026-06-15。

## 论文信息

- 论文：Sun, Ho and Lee, ICLR 2020 / arXiv:1909.03329
- 链接：https://openreview.net/forum?id=Skgxcn4YDS
- 引用：`@inproceedings{sun2020lamol, title={LAMOL: LAnguage MOdeling for Lifelong Language Learning}, author={Sun, Fan-Keng and Ho, Cheng-Hao and Lee, Hung-Yi}, booktitle={ICLR}, year={2020}}`

## 代码来源与下载状态

- 仓库：https://github.com/jojotenya/LAMOL.git
- Commit：03c31d9f0c7bf71295bc2d362ddf40a7656956e1
- 许可证/引用信息：MIT License（本地 LICENSE 已读取）。
- 下载记录：成功下载到 external_baselines/lamol，commit 03c31d9f0c7bf71295bc2d362ddf40a7656956e1。

## Method 原理

同一个 LM 同时学习任务求解和旧任务样本生成；新任务前生成旧任务伪样本，与新数据混合训练。

## 论文实验设置

官方 README 使用 GPT-2/openai-gpt，任务包括 SQuAD、IWSLT、CNN/DM、SST、WOZ 等；示例 ./train.sh --seq_train_type lll --tasks sst srl woz.en。

## 依赖

GPUtil==1.4.0, pytorch-transformers==1.2.0, torch>=1.2.0；需要 Google Drive 数据和 env 中 DATA_DIR/MODEL_ROOT_DIR。

## 复现命令

- 本项目 smoke / scaffold：py_compile 通过；python external_baselines/lamol/train.py --help 失败：ModuleNotFoundError: No module named pytorch_transformers。未安装旧依赖，未启动训练。
- 论文或完整复跑：当前机器不能直接复跑论文结果：旧依赖未安装、数据需 Google Drive、需要设置 env 和模型输出目录。完整命令示例：cd external_baselines/lamol && cp env.example env && 编辑 DATA_DIR/MODEL_ROOT_DIR && ./train.sh --seq_train_type lll --tasks sst srl woz.en。

## 与本项目接口对接方案

可把本项目 instruction triples 转换为 LAMOL solver/generator 双格式；若走 Llama，需要实现伪样本生成和 replay ratio。

## 当前状态

downloaded + smoke partially blocked
