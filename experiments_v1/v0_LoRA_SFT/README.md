# v0_LoRA_SFT：顺序 LoRA SFT 基线

v1 对照起点。fresh LoRA 顺序 SFT（seed 123，r=16 / α=32 / q,v_proj，lr 2e-4，bs 2，每段 1 epoch，EOS mask false）。

- **指标**：见 [results/README.md](results/README.md)
- **全量 pred/gold CSV**（greedy，与表同一评估口径，不入库）：
  - `predictions/instrdialog_pred_gold.csv`（211 条）
  - `predictions/instrdialogpp_pred_gold.csv`（464 条）
