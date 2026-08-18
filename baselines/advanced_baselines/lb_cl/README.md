# LB-CL: Learn More but Bother Less Continual Learning

状态：faithful scaffold + local smoke-tested。
更新日期：2026-06-15。

## 论文信息

- 论文：Qiao and Mahdavi, NeurIPS 2024
- 链接：https://openreview.net/forum?id=ZxtaNh5UYB
- 引用：`@inproceedings{qiao2024learn, title={Learn more, but bother less: parameter efficient continual learning}, author={Qiao, Fuli and Mahdavi, Mehrdad}, booktitle={NeurIPS}, year={2024}}`

## 代码来源与下载状态

- 仓库：未找到官方公开代码仓库
- Commit：无
- 许可证/引用信息：无代码许可证；OpenReview 页面显示论文 CC BY 4.0。
- 下载记录：全网搜索仅确认 OpenReview/NeurIPS PDF/Slides，未找到官方仓库；未伪造下载状态。

## Method 原理

对旧任务低秩参数做 SVD triplet 敏感度分析，选择高价值 triplet 初始化新任务低秩参数，并通过梯度投影让新任务训练避开旧低秩子空间。

## 论文实验设置

论文在标准 LLM continual learning benchmarks 上比较 O-LoRA 等方法，重点报告 forgetting、平均任务性能和 unseen/generalization 能力。

## 依赖

无官方依赖；本项目 faithful reimplementation 预计依赖 torch/peft/transformers，并需要访问 LoRA 权重、SVD 和梯度 hook。

## 复现命令

- 本项目 smoke：通过：`python core/train.py --config configs/baselines/smoke_lb_cl.yaml`；2 个 mock segment，结果写入 `results/logs/smoke_lb_cl.log` 和 `results/tables/smoke_lb_cl_segment_metrics.csv`。
- 当前实现：已记录 compact SVD triplet summary，并调用可验证 gradient projection hook。
- 论文或完整复跑：当前机器不能直接复跑论文结果：无官方代码，仍需实现论文级 triplet sensitivity 排序、triplet injection 初始化策略，并准备论文 benchmark/模型。

## 与本项目接口对接方案

新增 `baseline_name=lb_cl`、`configs/baselines/lb_cl.yaml` 和 `baselines/advanced_baselines/lb_cl/method.py`；当前已从占位缓存推进到 SVD triplet summary + projection hook，后续再补完整敏感度评分和初始化注入。

## 当前状态

faithful scaffold + local smoke-tested
