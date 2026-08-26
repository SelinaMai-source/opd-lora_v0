# v1_teacher_enhance_OPSD：teacher prompt 格式约束 + SFT 初始化 OPSD

## 设置

- **方法**：与 `v1_SFT_OPSD` 相同（加载 v0 SFT `final_adapter`，再做 on-policy JSD OPSD）。唯一改动是 teacher 过渡指令后追加格式约束：
  `Preserve its key content, answer format, wording style, and approximate length.`
- **开关**：`opsd.teacher_format_constraint: true`（默认旧 OPSD 配置为 false，本变体开启）
- **baseline_name**：`opsd`
- **初始化**：`results/runs/v0_sft_instrdialog/final_adapter` 与 `results/runs/v0_sft_instrdialogpp/final_adapter`
- **其余口径**：seed 123，r=16 / α=32 / dropout 0.05 / q,v_proj，lr 2e-4，bs 2，每段 1 epoch，EOS mask false，Llama-3.1-8B-Instruct bfloat16
- **配置**：`configs/` 内为本变体 yaml 副本（与主 `configs/` 一致）

正式结果见 [results/README.md](results/README.md)。两条流均完成；格式约束相对无约束 SFT+OPSD **负向**（InstrDialog Score 0.153，++ 0.244）。

## 复现（勿在 GPU 被占用时跑）

```bash
cd /root/autodl-tmp/rp_lora_v0 && unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 冒烟（InstrDialog 2 段）
python -m core.train --config configs/smoke_v1_teacher_enhance_opsd.yaml
# 正式两条流
python -m core.train --config configs/v1_teacher_enhance_opsd_instrdialog.yaml
python -m core.train --config configs/v1_teacher_enhance_opsd_instrdialogpp.yaml
```
