# Lit_SDFT

ICML 2026 *Self-Distillation Enables Continual Learning*.

官方 README 写 `Continual-Intelligence/Self-Distillation`，该 URL **404**。作者公开仓：https://github.com/idanshen/Self-Distillation（clone 到 `official/`，见 `official/CLONE_NOTE.txt`）。

接入：student 每例 1 条短 on-policy rollout（≤32 tokens）；teacher = 同一 LoRA + demonstration-conditioned prompt（stop-grad）；KL。复用 `run_opsd_step(teacher_source=adapter)`。不跑官方 TRL-H200 trainer。seed 456。

- yaml：`configs/lit_sdft_*_seed456.yaml`
- `baseline_name: sdft`
