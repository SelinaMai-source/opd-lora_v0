# Lit_CLoRA

ACL 2025. Official: https://github.com/sutakori/CLoRA (`official/clora.py`).

接入：`CLoRAMethod` 在 CE 上加官方 subspace regularizer（冻结正交 U/V，λ||A V||_F² + ||Bᵀ U||_F²）。不跑官方 trainer。k=128（官方默认 512，4090 上缩小）。seed 456。

- yaml：`configs/lit_clora_instrdialog_seed456.yaml`、`*_instrdialogpp_*`、`*_smoke2seg.yaml`
- `baseline_name: clora`
