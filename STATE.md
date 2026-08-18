# RP-LoRA v0 进度存档(2026-08-17 22:25)

## 当前状态:v0 baseline 已完成,指标正常
- [x] rp_lora_v0 目录已组装:core/(v3 完整栈)、baselines/(commit d711d91 旧布局)、data/processed/(instrdialog 19 段 + instrdialog++ 38 段,与 v3 同源)
- [x] conda 环境就绪:/root/autodl-tmp/conda_envs/rp_lora_v0(python 3.10, torch 2.8.0+cu128, transformers 4.48.3, peft 0.20.0)
- [x] 基座模型:assets/pretrained/Meta-Llama-3.1-8B-Instruct(16.06G,校验通过,确认为 Instruct 版)
- [x] GPU:RTX 4090 48G 可用
- [x] **发现并修复关键训练 bug:EOS 从未被监督**(core/models/base_model.py 默认 mask_eos_token_in_labels=True 且 mask_all_special_tokens_in_labels=True,Llama-3.1 的 <|eot_id|>(128009) 同时是 eos 和 special token 被双重 mask,模型答完不停)。修复方式:configs 的 model 段加 `mask_eos_token_in_labels: false` + `mask_all_special_tokens_in_labels: false`(三个 v0 config 均已改,纯配置改动,代码未动)
- [x] 两条数据流正式训练完成(EOS 修复版,seed 123,LoRA r=16/alpha=32/q,v_proj,lr 2e-4,bs 2,每段 1 epoch)
- [x] 指标判定正常:strict EM 约 31%,远高于旧 Base 模型的 4%,两条流结果一致

## v0 最终指标(EOS 修复版,评估口径与 v3 一致)

| 指标 | instrdialog(19 段)修复前 | instrdialog(19 段)修复后 | instrdialog++(38 段)修复后 |
|---|---|---|---|
| seen_avg_score (strict EM) | 0.0105 | **0.3096** | **0.3135** |
| seen_avg_task_aware_score | 0.2307 | 0.3254 | 0.3266 |
| forgetting | 0.233 | 0.133 | 0.156 |
| task_aware_forgetting | 0.211 | 0.128 | 0.154 |
| token_f1 | 0.090 | 0.393 | 0.411 |
| rouge_l | 0.087 | 0.388 | 0.404 |
| bleu | 0.114 | 0.424 | 0.472 |

- 修复版运行 ID:instrdialog = `20260817_213117_v0_sequential_instrdialog`,instrdialog++ = `20260817_215606_v0_sequential_instrdialogpp`
- 旧(修复前)instrdialog 运行 `20260817_174950_v0_sequential_instrdialog` 保留作对照;另有 2 个被中止/OOM 的部分运行目录可忽略

## 过程中的关键事件(备查)
1. 旧配置 instrdialog strict EM 仅 1.05% → 诊断结论:主因是 EOS 不监督 + 灾难性遗忘 + 长文本任务占比高,评估抽取问题 ≤3-4 个百分点。诊断证据在 eval_debug dump。
2. instrdialog++ 修复版首次运行在第 27 段(cnn_dailymail 长文档摘要)CUDA OOM(峰值缺口约 1.2G)→ 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后重跑成功。若后续更大模型/更长序列仍 OOM,备选方案是给 backbone 加梯度检查点(不改训练数学)。
3. 后台训练建议用 nohup/setsid 脱离终端会话,避免会话结束连带终止训练进程。

## 关键命令(重跑/复现)
```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # instrdialog++ 长文档段需要
/root/autodl-tmp/conda_envs/rp_lora_v0/bin/python -m core.train --config configs/v0_sequential_instrdialog.yaml
/root/autodl-tmp/conda_envs/rp_lora_v0/bin/python -m core.train --config configs/v0_sequential_instrdialogpp.yaml
```

## 环境注意
- shell 环境里 CUDA_VISIBLE_DEVICES 为空字符串,运行训练前必须 `unset CUDA_VISIBLE_DEVICES`。

## 下一步
v0 baseline 指标已正常(约 31% strict EM / 33% task-aware,遗忘 0.13-0.16),可基于它讨论 v1 设计。
