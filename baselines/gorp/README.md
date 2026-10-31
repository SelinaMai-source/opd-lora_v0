# Lit_GORP

ACL 2025. Official: https://github.com/Wcxwcxw/GORP

接入：对照 `src/Proj_torch/proj_projector.py` 的 std 投影。CE backward 后把每个 LoRA 矩阵梯度投到 rank=8 SVD 子空间再投影回去。不跑官方 SuperNI/TRACE trainer。seed 456。

- yaml：`configs/lit_gorp_*_seed456.yaml`
- `baseline_name: gorp`
