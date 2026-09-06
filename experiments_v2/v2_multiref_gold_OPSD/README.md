# v2_multiref_gold_OPSD：多参考合成 + 同一步 CE+λ OPSD

## 设置

- **方法**：CE 从 `ex.outputs` 随机抽一个 gold；OPSD 对学生 rollout 在多个 gold 里选 Token F1 最高者（ROUGE-L 平手）交给 teacher。`L = L_CE + λ L_OPSD`。teacher 默认开格式约束。≥1024 token 只留 CE。
- **baseline_name**：`ce_opsd`
- **λ**：{0.1, 0.2, 0.3}（`ce_opsd.lambda_opsd`）
- **数据**：`benchmark/.../*_multiref.json`（按 instruction+input 合并多参考，不覆盖 CITB 旧文件）
- **实现**：`baselines/basic_baselines/ce_opsd/method.py`（本目录拷贝：`../methods/ce_opsd/method.py`）

## 复现（勿在 GPU 被占用时跑）

```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m core.train --config configs/v2_multiref_gold_opsd_instrdialog_l02_seed456_smoke2seg.yaml
python -m core.train --config configs/v2_multiref_gold_opsd_instrdialog_l02_seed456.yaml
python -m core.train --config configs/v2_multiref_gold_opsd_instrdialogpp_l02_seed456.yaml
```

正式结果见 [results/README.md](results/README.md)。
