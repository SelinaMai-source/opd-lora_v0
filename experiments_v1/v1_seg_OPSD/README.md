# v1_seg_OPSD：段内 SFT / OPSD 交替 + 冻结 LoRA teacher

## 设置

- **方法**：每个 CL segment 内按 K 个 optimizer step 切块：SFT（`fit_batch` CE）K 步 → 把当前 `default` LoRA 拷到冻结 adapter `teacher` → OPSD K 步（teacher 用该冻结 LoRA，**不是** `disable_adapter` 基座）→ 再 SFT… 直到该段数据耗尽。
- **baseline_name**：`seg_opsd`
- **K 网格**：`{25, 50, 75}`（约 1/2/3 epoch/phase；train50 / bs2 ⇒ 1 epoch ≈ 25 step）
- **epochs**：为完成至少一轮 SFT→OPSD，K=25 用 2 epoch，K=50 用 4，K=75 用 6。InstrDialog++ 默认 K=25 / 2 epoch，等 InstrDialog 网格胜出后再改。
- **初始化**：fresh LoRA（不加载 SFT adapter）
- **格式约束**：关闭（`opsd.teacher_format_constraint: false`），与 v1_SFT_OPSD teacher 文案对齐，只测交替与冻结 LoRA teacher
- **其余口径**：seed 123，r=16 / α=32 / dropout 0.05 / q,v_proj，lr 2e-4，bs 2，EOS mask false
- **实现**：`baselines/basic_baselines/seg_opsd/method.py`；拷贝权重见 `LoRAWrapper.copy_adapter_weights`

正式结果见 [results/README.md](results/README.md)。InstrDialog 网格胜出 **K=25**（Score 0.289 / F 0.156）；++ 上 Score 0.289 与 v0 持平、F 0.151 优于 v0 的 0.180。

## 复现（勿在 GPU 被占用时跑）

```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 冒烟：K=2、InstrDialog 2 段（8 train / 4 eval），保证 SFT 块与 OPSD 块都跑到
python -m core.train --config configs/smoke_v1_seg_opsd.yaml
# InstrDialog 网格
python -m core.train --config configs/v1_seg_opsd_instrdialog_k25.yaml
python -m core.train --config configs/v1_seg_opsd_instrdialog_k50.yaml
python -m core.train --config configs/v1_seg_opsd_instrdialog_k75.yaml
# 胜出 K 打 InstrDialog++（当前默认 K=25）
python -m core.train --config configs/v1_seg_opsd_instrdialogpp.yaml
```
