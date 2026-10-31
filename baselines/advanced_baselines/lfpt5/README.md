# LFPT5: Lifelong Few-shot Language Learning with Prompt Tuning of T5

状态：blocked。
更新日期：2026-06-15。

## 论文信息

- 论文：Qin and Joty, ICLR 2022 / arXiv:2110.07298
- 链接：https://openreview.net/forum?id=HCRVf71PMF
- 引用：`@inproceedings{qin2022lfpt5, title={LFPT5: A Unified Framework for Lifelong Few-shot Language Learning Based on Prompt Tuning of T5}, author={Qin, Chengwei and Joty, Shafiq}, booktitle={ICLR}, year={2022}}`

## 代码来源与下载状态

- 仓库：https://github.com/qcwthu/Lifelong-Fewshot-Language-Learning.git
- Commit：无有效本地 commit
- 许可证/引用信息：未能下载 LICENSE，待网络恢复后复核。
- 下载记录：git clone 失败：GnuTLS recv error (-110)；git ls-remote exit=124。

## Method 原理

冻结 T5 backbone，训练 task/domain soft prompt，同时让模型作为 solver 和 generator，生成旧域伪样本并用 KL consistency 缓解遗忘。

## 论文实验设置

官方 README 要求下载 LM-adapted T5-large TF checkpoint、转换到 PyTorch，并修改 lm_adapted_path/cache_path 后运行 DiffType/SameType 脚本。

## 依赖

官方依赖未能读取；需要 LM-adapted T5-large、transformers、datasets/cache 和论文 few-shot 数据。

## 复现命令

- 本项目 smoke / scaffold：未运行官方 smoke；仓库未完整下载。本项目尚未注册 lfpt5 入口。
- 论文或完整复跑：当前机器不能直接复跑论文结果：官方代码未下载，T5-large checkpoint/转换步骤和数据未准备。

## 与本项目接口对接方案

建议先实现 prefix/prompt tuning + self-generated replay，并明确 T5 encoder-decoder 与本项目 Llama causal LM 的差异。

## 当前状态

blocked
