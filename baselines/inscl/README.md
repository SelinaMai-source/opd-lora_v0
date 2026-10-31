# Lit_InsCL

NAACL 2024. Official: https://github.com/OPPO-Mente-Lab/InsCL

接入：对照 `official/replay/InsCL_sampling.py`。instruction embedding 1-Wasserstein 分配 replay 配额，InsInfo 选样。无 GPT-4 tag，用 length×diversity 代理。K=10。seed 456。不跑官方 Llama-7B trainer。

- yaml：`configs/lit_inscl_*_seed456.yaml`
- `baseline_name: inscl`
