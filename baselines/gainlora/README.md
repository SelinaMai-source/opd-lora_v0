# Lit_GainLoRA

NeurIPS 2025. Official: https://github.com/liangyanshuo/gainlora

接入：每段新建 LoRA；eval 用 `set_soft_routing`，旧任务上把新 LoRA gate 压到 0.05。不跑官方 T5/Llama SuperNI 脚本。seed 456。

- yaml：`configs/lit_gainlora_*_seed456.yaml`
- `baseline_name: gainlora`
