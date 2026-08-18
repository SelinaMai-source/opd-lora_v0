# RP-LoRA v0:朴素 baseline

Llama-3.1-8B-Instruct + LoRA 在持续学习数据流上的朴素 sequential 微调 baseline(无 replay / 无多分支 / 无路由)。

## v0 最终指标(seed 123,LoRA r=16/alpha=32 作用于 q,v_proj,lr 2e-4,bs 2,每段 1 epoch)

| 指标 | instrdialog(19 段) | instrdialog++(38 段) |
|---|---|---|
| seen_avg_score (strict EM) | 0.3096 | 0.3135 |
| seen_avg_task_aware_score | 0.3254 | 0.3266 |
| forgetting | 0.133 | 0.156 |
| token_f1 | 0.393 | 0.411 |

完整口径、修复记录与运行 ID 见 [STATE.md](STATE.md)。

## 目录结构

- `core/` — 训练/评估主栈(v3 同源)
- `configs/` — v0 配置(含 EOS 监督修复:`mask_eos_token_in_labels: false`、`mask_all_special_tokens_in_labels: false`)
- `data/processed/` — 两条数据流(train50/eval10)
- `baselines/` — 旧布局基线参考实现(commit d711d91)
- `results/` — 全部运行产物(指标、日志、逐样本 eval_debug)
- `assets/` — 模型权重,不入库,获取方式见下

## 模型获取

```bash
pip install modelscope
modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct --local_dir assets/pretrained/Meta-Llama-3.1-8B-Instruct
```

## 复现

```bash
unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m core.train --config configs/v0_sequential_instrdialog.yaml
python -m core.train --config configs/v0_sequential_instrdialogpp.yaml
```
