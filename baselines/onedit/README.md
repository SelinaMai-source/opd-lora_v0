# Lit_OnEDIT

AAAI 2026. **无公开官方仓。** 禁止伪造 official clone。

论文复现：固定 4 个 implicit-task adapter，无 task-id。段开始按 instruction embedding 匹配或开新槽（容量满则最近邻）。只训当前槽。eval 按 centroid 路由。本 setting 仍按段边界喂数据。seed 456。

- yaml：`configs/lit_onedit_*_seed456.yaml`
- `baseline_name: onedit`
- 论文：https://ojs.aaai.org/index.php/AAAI/article/view/39440
