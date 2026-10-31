# Lit_KPIG

ACL 2025 *Don't Half-listen*. **无官方公开仓。** 禁止伪造 official clone。

论文复现：mask instruction 最长 token 作 key-part，IG = CE(full)−CE(masked)，回放最低 IG 的 K=10 条；训练 CE + λ masked JSD（相对上一段 LoRA 快照）。seed 456。

- yaml：`configs/lit_kpig_*_seed456.yaml`
- `baseline_name: kpig`
- 论文：https://aclanthology.org/2025.acl-long.1153/
