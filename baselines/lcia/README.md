# LCIA: Large Continual Instruction Assistant

状态：已接入本仓库统一训练循环（文本-only，Llama-3.1-8B-Instruct + LoRA）。本轮只做代码/配置/冒烟准备，**未跑 19/38 段正式训练**。
更新日期：2026-08-28。

## 论文信息

- 论文：Qiao et al., ICML 2025 / arXiv:2410.10868
- 链接：https://arxiv.org/abs/2410.10868
- 引用：`@inproceedings{qiao2025lcia, title={Large Continual Instruction Assistant}, author={Qiao, Jingyang and others}, booktitle={ICML}, year={2025}}`
- 官方代码：https://github.com/JingyangQiao/CoIN （`instruct/LoRASelect.py` 指令分组，`CoIN/peft/tuners/lora.py` 动态 EMA）

## Method 原理

两块，对照论文与 CoIN 实现，接到本项目 `LoRAWrapper`：

1. **指令分组**：段开始时收集该段 unique instruction，用 TF-IDF（sklearn `TfidfVectorizer`，无 sklearn 时走本地 fallback）与 codebook 各组做平均 pairwise 余弦。相似度 ≥ `epsilon`（默认 0.90，对齐 CoIN 的 Instructions 模板阈值）则复用该组 LoRA，否则新建 adapter。复用时把未见过的 instruction 追加进该组。
2. **动态 EMA**：每个 optimizer step 之后，按 LoRA 模块做 L1 近似得到新参数权重 α（越界则 α=0.01 ⇔ β=0.99），`θ*_t = β θ*_{t-1} + (1-β) θ_t`。段结束把 EMA 参数写回 adapter，评测与后续复用都走这份权重。
3. **评测路由**：按 instruction / prompt 的 TF-IDF 相似度选组；优先 substring 命中 codebook 里的 instruction。`evaluate_stream` 只对 `router_only` 做逐条路由，因此 LCIA 在 `on_segment_start` 包装 `model.generate` / `generate_with_ids`，不改 `evaluate.py`。

文本-only，不接视觉 projector。

## 本项目设定

与 v0 Sequential LoRA 对齐：**Llama-3.1-8B-Instruct**、LoRA r=16 / alpha=32 / `q_proj,v_proj`、lr 2e-4、bs=2、每段 1 epoch、**seed 456**、EOS mask flags false。

## 配置

- 正式：`configs/v2pre_lcia_instrdialog_seed456.yaml`、`configs/v2pre_lcia_instrdialogpp_seed456.yaml`
- 2 段冒烟：`configs/smoke_v2pre_lcia_instrdialog.yaml`、`configs/smoke_v2pre_lcia_instrdialogpp.yaml`

`processed_stream_dir` 已指向 `benchmark/instrdialog` 与 `benchmark/instrdialogpp`（与 v0 相同 json 文件名）。若该目录不存在，可改回 `data/processed`。

## 复现命令（稍后 GPU 空闲再跑，本轮不要开全流）

InstrDialog 2 段冒烟：

```bash
cd /root/autodl-tmp/rp_lora_v0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python core/train.py --config configs/smoke_v2pre_lcia_instrdialog.yaml
```

正式短流 / 长流（19 段、38 段）：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python core/train.py --config configs/v2pre_lcia_instrdialog_seed456.yaml

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python core/train.py --config configs/v2pre_lcia_instrdialogpp_seed456.yaml
```

计划中的 GPU 顺序：InstrDialog 冒烟 → CITB Sequential 短流 → LCIA 短流 → 两条长流。

## 与本项目接口对接方案

- `baseline_name: lcia`，在 `core/train.py` 的 `_build_baseline_method` 注册。
- 训练接口与 `SequentialLoRAMethod` / `OLoraMethod` 相同：`on_segment_start` + `train_on_segment(segment, model, lora, lr, epochs, batch_size)`。
- 多 adapter：组 0 复用 PEFT 的 `default`，之后为 `lcia_g1`, `lcia_g2`, …；训练时冻结非当前组。
- 超参写在 config 的 `lcia:` 段（也兼容 `advanced_baseline.lcia`）。
