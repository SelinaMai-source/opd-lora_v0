# Lit_SSR

ACL 2024. Official: https://github.com/XMUDeepLIT/SSR

接入：对照 `official/custom/icl_gen` 三步。base（disable_adapter）ICL 合成 → 当前 LoRA refine → hashed embedding 多样筛选，每段上限 K=10，混入 CE。不跑 LLaMA-Factory 官方脚本。seed 456。

- yaml：`configs/lit_ssr_*_seed456.yaml`
- `baseline_name: ssr`
