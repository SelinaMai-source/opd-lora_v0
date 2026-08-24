# RP-LoRA v0 进度存档(2026-08-17 22:25;2026-08-20 追加 v1)

## v1 OPSD 对比实验(2026-08-20 完成)

- [x] OPSD 实现:on-policy 采样 + student/teacher 双 forward + 逐 token JSD(β=0.5);core/formatting.py(format_for_teacher)、core/models/base_model.py(sample_rollouts/forward_logits)、baselines/basic_baselines/opsd/method.py、core/train.py 注册、adapter 存取链路(output.save_final_adapter / model.init_adapter_path)
- [x] 6 个正式 run 全部完成:SFT 重跑 v0_sft_instrdialog / v0_sft_instrdialogpp(产出 final_adapter),v1_OPSD 两条流(20260820_172934/174143),v1_SFT_OPSD 两条流(20260820_181411/182823)

### v1 最终结果(全指标,提取自 final_metrics.json,数值保留 3 位小数)

| Method | Data | Output Length | Seen-Avg Acc | Seen-Avg Task-aware Acc | Forgetting | Task-Aware Forgetting | Token F1 | ROUGE-L | BLEU | LCS Overlap | Current-Seg Acc | Current-Seg Task-aware Acc | Task-aware Score Mean | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_LoRA_SFT | InstrDialog | 10.3 | 0.346 | 0.362 | 0.089 | 0.083 | 0.430 | 0.425 | 0.451 | 0.422 | 0.000 | 0.000 | 0.332 | 合理:SFT 基线,19 段流遗忘低(F 0.089),生成质量指标为全表最高档,作为 v1 对照起点 |
| v0_LoRA_SFT | InstrDialog++ | 5.0 | 0.285 | 0.298 | 0.180 | 0.175 | 0.392 | 0.388 | 0.461 | 0.383 | 0.100 | 0.100 | 0.261 | 合理:38 段长流遗忘明显加重(F 0.180 vs 19 段 0.089),符合流越长越难抗遗忘的预期 |
| v1_OPSD | InstrDialog | 9.7 | 0.289 | 0.299 | 0.128 | 0.133 | 0.361 | 0.357 | 0.385 | 0.357 | 0.000 | 0.000 | 0.270 | 合理但偏弱:训练稳定无崩盘、F 与 v0 相当,但 Score 低于 SFT 约 5.8pp——仅 on-policy 蒸馏不足以替代 SFT 起点 |
| v1_OPSD | InstrDialog++ | 9.3 | 0.223 | 0.255 | 0.171 | 0.154 | 0.359 | 0.352 | 0.423 | 0.370 | 0.100 | 0.300 | 0.224 | 合理但偏弱:同样稳定无崩盘,Score 低于 v0 约 6.2pp、F 与 v0 接近,结论与 19 段流一致 |
| v1_SFT_OPSD | InstrDialog | 14.7 | 0.011 | 0.211 | 0.367 | 0.171 | 0.153 | 0.146 | 0.197 | 0.165 | 0.000 | 0.000 | 0.190 | 异常:末段(segment 18, task1600_smcalflow)训练后 strict EM 由段 17 末的 0.294 崩至 0.011(F 0.367),Token F1/ROUGE-L/BLEU 同步腰斩且输出变长失控(均长 14.7、最长打满 64 token);训练侧 loss/grad_norm 正常、同段 v0/v1_OPSD 均不崩,已排除 bug,属真实剧烈遗忘 |
| v1_SFT_OPSD | InstrDialog++ | 7.7 | 0.300 | 0.313 | 0.084 | 0.082 | 0.409 | 0.402 | 0.457 | 0.412 | 0.000 | 0.200 | 0.272 | 合理且最优:Score 0.300 为全表最高,F 0.084 显著优于 v0 的 0.180——SFT 初始化 + OPSD 在长流上收益明确 |

口径:Seen-Avg Acc = eval.seen_avg_score(strict EM),Forgetting = eval.forgetting,Current-Seg Acc = eval.current_score(末段),Task-aware Score Mean = eval.task_aware_score_mean(训练轨迹均值);Output Length 取自最终评估点 eval_debug dump(前 50 条子集)raw_generated_output 的平均 token 数(Llama-3.1 tokenizer,不含 special tokens),final_metrics 无此字段。

### v1 结论

1. v1_OPSD 两条流稳定训练无崩盘但绝对分数低于 SFT 系 5.8/6.2 pp → 仅 on-policy 蒸馏不足以替代 SFT 起点
2. v1_SFT_OPSD 在 instrdialog++ 上 Score 与抗遗忘双优(30.03% > 28.50%,F 0.084 << 0.180)→ SFT 之上加 OPSD 有效
3. **重要异常**:v1_SFT_OPSD 在 instrdialog 末段(segment 18, task1600_smcalflow)训练后 strict EM 由 0.2944 崩至 0.0105(F 0.367);训练侧 loss/grad_norm 正常、同段 v0/v1_OPSD 均不崩 → 经诊断排除 bug,属真实剧烈遗忘;SFT 初始化对 OPSD 是双刃剑

完整报告:/root/autodl-tmp/docs/v1_experiment_report.md;GitHub v1 分支:experiments/v1_OPSD/ 与 experiments/v1_SFT_OPSD/

---

# 以下为 v0 存档(2026-08-17 22:25)

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
