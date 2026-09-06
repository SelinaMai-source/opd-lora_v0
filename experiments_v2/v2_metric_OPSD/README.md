# v2_metric_OPSD：开放生成 ranking + 闭式格式负样本

## 设置

- **方法**：建在 multiref 骨干上（λ=0.2，先不加 proto）。每隔 6 个 batch：对学生采 4 个候选，分数 = Token F1 + ROUGE-L + BLEU（对抽中的 gold），pairwise 推高分压低分。闭式任务（`label_accuracy` / `after_step_extracted_em`）走 neg 模板负样本。≥1024 token 跳过 ranking。
- **baseline_name**：`metric_opsd`
- **μ**：{0.05, 0.1, 0.2}
- **实现**：`baselines/basic_baselines/metric_opsd/method.py`

对照：同 λ=0.2 的 `v2_multiref`。

## 复现

```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m core.train --config configs/v2_metric_opsd_instrdialog_m01_seed456.yaml
python -m core.train --config configs/v2_metric_opsd_instrdialogpp_m01_seed456.yaml
```

正式结果见 [results/README.md](results/README.md)。
