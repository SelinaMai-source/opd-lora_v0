# Lit_SAPT

ACL 2024. Official: https://github.com/circle-hit/SAPT

接入：每段 LoRA。SALS = instruction centroid 余弦 softmax 作为 `set_soft_routing`。ARM = 对旧 prompt 生成 K=10 伪样本混入 CE。eval 走同一 attention 路由。不跑官方 T5 trainer。seed 456。

- yaml：`configs/lit_sapt_*_seed456.yaml`
- `baseline_name: sapt`
