# Lit_TreeLoRA

ICML 2025. Official: https://github.com/ZinYY/TreeLoRA

接入：每段 LoRA；q_proj/v_proj 上 LoRA-A 梯度做树选择，对选中的旧任务梯度加负点积正则（对照 `official/utils/kd_lora_tree.py`）。**不整仓 DeepSpeed 重跑。** eval oracle 分段 adapter。seed 456。

- yaml：`configs/lit_treelora_*_seed456.yaml`
- `baseline_name: treelora`
