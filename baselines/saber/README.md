# Lit_SABER

ICML 2026. Official: https://github.com/OptMN-Lab/SABER-ICML-2026

接入：无 replay。每段一条 LoRA。`proj_cos` 选相关旧 adapter，梯度投影掉 SVD 保护基后做 backward refine（对照官方 `_safe_backward_step`）。eval 按 source_segment_id oracle 路由。不跑官方 prompt trainer。seed 456。

- yaml：`configs/lit_saber_*_seed456.yaml`
- `baseline_name: saber`
