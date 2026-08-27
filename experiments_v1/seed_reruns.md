# P0+P1 双 seed 重跑清单

共 14 次：teacher_enhance / seg_OPSD_replay / seg_OPSD K=25 / SFT_OPSD++。
原 run 未覆盖（固定 `run_name`）。SFT init 变体两边 seed 共用同一份 v0 SFT adapter。

| # | 方法 | Data | seed | 内容 | run_name | run 目录 | seen_avg↑ | F↓ | token_f1↑ | rouge_l↑ | bleu↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | v1_teacher_enhance_OPSD | InstrDialog | 123 | seed 123 重跑 | `v1_teacher_enhance_opsd_instrdialog_seed123_rerun` | `results/runs/v1_teacher_enhance_opsd_instrdialog_seed123_rerun` | 0.253 | 0.127 | 0.332 | 0.327 | 0.376 |
| 2 | v1_teacher_enhance_OPSD | InstrDialog | 456 | seed 456 | `v1_teacher_enhance_opsd_instrdialog_seed456` | `results/runs/v1_teacher_enhance_opsd_instrdialog_seed456` | 0.325 | 0.033 | 0.396 | 0.390 | 0.426 |
| 3 | v1_teacher_enhance_OPSD | InstrDialog++ | 123 | seed 123 重跑 | `v1_teacher_enhance_opsd_instrdialogpp_seed123_rerun` | `results/runs/v1_teacher_enhance_opsd_instrdialogpp_seed123_rerun` | 0.274 | 0.089 | 0.401 | 0.394 | 0.438 |
| 4 | v1_teacher_enhance_OPSD | InstrDialog++ | 456 | seed 456 | `v1_teacher_enhance_opsd_instrdialogpp_seed456` | `results/runs/v1_teacher_enhance_opsd_instrdialogpp_seed456` | 0.263 | 0.108 | 0.401 | 0.394 | 0.444 |
| 5 | v1_seg_OPSD_replay K=25 | InstrDialog | 123 | seed 123 重跑 | `v1_seg_opsd_replay_instrdialog_seed123_rerun` | `results/runs/v1_seg_opsd_replay_instrdialog_seed123_rerun` | 0.357 | 0.072 | 0.419 | 0.416 | 0.454 |
| 6 | v1_seg_OPSD_replay K=25 | InstrDialog | 456 | seed 456 | `v1_seg_opsd_replay_instrdialog_seed456` | `results/runs/v1_seg_opsd_replay_instrdialog_seed456` | 0.331 | 0.135 | 0.414 | 0.409 | 0.456 |
| 7 | v1_seg_OPSD_replay K=25 | InstrDialog++ | 123 | seed 123 重跑 | `v1_seg_opsd_replay_instrdialogpp_seed123_rerun` | `results/runs/v1_seg_opsd_replay_instrdialogpp_seed123_rerun` | 0.387 | 0.108 | 0.490 | 0.484 | 0.526 |
| 8 | v1_seg_OPSD_replay K=25 | InstrDialog++ | 456 | seed 456 | `v1_seg_opsd_replay_instrdialogpp_seed456` | `results/runs/v1_seg_opsd_replay_instrdialogpp_seed456` | 0.416 | 0.066 | 0.497 | 0.491 | 0.531 |
| 9 | v1_seg_OPSD K=25 | InstrDialog | 123 | seed 123 重跑 | `v1_seg_opsd_k25_instrdialog_seed123_rerun` | `results/runs/v1_seg_opsd_k25_instrdialog_seed123_rerun` | 0.273 | 0.178 | 0.346 | 0.342 | 0.391 |
| 10 | v1_seg_OPSD K=25 | InstrDialog | 456 | seed 456 | `v1_seg_opsd_k25_instrdialog_seed456` | `results/runs/v1_seg_opsd_k25_instrdialog_seed456` | 0.005 | 0.421 | 0.126 | 0.122 | 0.187 |
| 11 | v1_seg_OPSD K=25 | InstrDialog++ | 123 | seed 123 重跑 | `v1_seg_opsd_k25_instrdialogpp_seed123_rerun` | `results/runs/v1_seg_opsd_k25_instrdialogpp_seed123_rerun` | 0.316 | 0.159 | 0.417 | 0.409 | 0.471 |
| 12 | v1_seg_OPSD K=25 | InstrDialog++ | 456 | seed 456 | `v1_seg_opsd_k25_instrdialogpp_seed456` | `results/runs/v1_seg_opsd_k25_instrdialogpp_seed456` | 0.263 | 0.187 | 0.379 | 0.373 | 0.440 |
| 13 | v1_SFT_OPSD | InstrDialog++ | 123 | seed 123 重跑 | `v1_sft_opsd_instrdialogpp_seed123_rerun` | `results/runs/v1_sft_opsd_instrdialogpp_seed123_rerun` | 0.287 | 0.111 | 0.411 | 0.405 | 0.464 |
| 14 | v1_SFT_OPSD | InstrDialog++ | 456 | seed 456 | `v1_sft_opsd_instrdialogpp_seed456` | `results/runs/v1_sft_opsd_instrdialogpp_seed456` | 0.268 | 0.114 | 0.404 | 0.398 | 0.448 |

完成 14/14。指标提取自各 run 的 `final_metrics.json`，保留 3 位小数。

确认重跑 4/4 完成（不覆盖上表 14 行，另开 run_name）。指标提取自 `results/runs/<run_name>/final_metrics.json`。

| # | 方法 | Data | seed | 内容 | run_name | run 目录 | seen_avg↑ | F↓ | token_f1↑ | rouge_l↑ | bleu↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | v1_teacher_enhance_OPSD | InstrDialog | 123 | seed123 rerun2 | `v1_teacher_enhance_opsd_instrdialog_seed123_rerun2` | `results/runs/v1_teacher_enhance_opsd_instrdialog_seed123_rerun2` | 0.258 | 0.094 | 0.342 | 0.336 | 0.377 |
| 16 | v1_teacher_enhance_OPSD | InstrDialog | 789 | seed789 | `v1_teacher_enhance_opsd_instrdialog_seed789` | `results/runs/v1_teacher_enhance_opsd_instrdialog_seed789` | 0.152 | 0.211 | 0.231 | 0.227 | 0.348 |
| 17 | v1_seg_OPSD K=25 | InstrDialog | 456 | seed456 rerun | `v1_seg_opsd_k25_instrdialog_seed456_rerun` | `results/runs/v1_seg_opsd_k25_instrdialog_seed456_rerun` | 0.220 | 0.233 | 0.312 | 0.308 | 0.358 |
| 18 | v1_seg_OPSD K=25 | InstrDialog | 789 | seed789 | `v1_seg_opsd_k25_instrdialog_seed789` | `results/runs/v1_seg_opsd_k25_instrdialog_seed789` | 0.325 | 0.111 | 0.394 | 0.390 | 0.432 |

## 飞书全表（历史行 + 14 重跑行 + 4 确认重跑）

历史行沿用 `STATE.md` / `v1_experiment_report.md` 第 10.1 节；原 0.011 行保留。14 重跑行与 4 确认重跑行追加在表末，内容列写明 seed。数值来自 `final.eval.*`，3 位小数。

| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | Current-Seg Acc↑ | Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog | 0.346 | 0.362 | 0.089 | 0.083 | 0.430 | 0.425 | 0.451 | 0.422 | 0.000 | 0.000 | 0.332 | 合理：SFT 基线，19 段流遗忘低（F 0.089），生成质量指标为全表最高档，作为 v1 对照起点 |
| v0_LoRA_SFT | fresh LoRA 顺序 SFT（v0 基线） | InstrDialog++ | 0.285 | 0.298 | 0.180 | 0.175 | 0.392 | 0.388 | 0.461 | 0.383 | 0.100 | 0.100 | 0.261 | 合理：38 段长流遗忘明显加重（F 0.180 vs 19 段 0.089），符合流越长越难抗遗忘的预期 |
| v1_OPSD | fresh LoRA 直接 OPSD，无 SFT 初始化 | InstrDialog | 0.289 | 0.299 | 0.128 | 0.133 | 0.361 | 0.357 | 0.385 | 0.357 | 0.000 | 0.000 | 0.270 | 合理但偏弱：训练稳定无崩盘、F 与 v0 相当，但 Score 低于 SFT 约 5.8pp——仅 on-policy 蒸馏不足以替代 SFT 起点 |
| v1_OPSD | fresh LoRA 直接 OPSD，无 SFT 初始化 | InstrDialog++ | 0.223 | 0.255 | 0.171 | 0.154 | 0.359 | 0.352 | 0.423 | 0.370 | 0.100 | 0.300 | 0.224 | 合理但偏弱：同样稳定无崩盘，Score 低于 v0 约 6.2pp、F 与 v0 接近，结论与 19 段流一致 |
| v1_SFT_OPSD | 加载 SFT 最终 adapter 初始化，继续 OPSD（原 run seed 123） | InstrDialog | 0.011 | 0.211 | 0.367 | 0.171 | 0.153 | 0.146 | 0.197 | 0.165 | 0.000 | 0.000 | 0.190 | 异常（单次）：末段 segment 18（task1600_smcalflow）训练后 strict EM 由段 17 末的 0.294 崩至 0.011（F 0.367）；2026-08-24 同 seed 重跑未崩、seed 456 亦未崩 → 属 GPU 非确定性 + seed 敏感临界点，非该流系统性失稳 |
| v1_SFT_OPSD | 同 seed 123 重跑（每段保存 adapter） | InstrDialog | 0.263 | 0.283 | 0.182 | 0.172 | 0.361 | 0.353 | 0.396 | 0.366 | 0.000 | 0.000 | 0.256 | 未崩：段 18 后 Score 0.263（相对段 17 的 0.210 回升），F 0.182；同 seed 无法复现原 0.011 崩盘，说明原崩溃含 GPU 非确定性成分 |
| v1_SFT_OPSD | seed 456 稳健性（同 SFT init adapter） | InstrDialog | 0.347 | 0.358 | 0.033 | 0.033 | 0.420 | 0.415 | 0.443 | 0.423 | 0.000 | 0.000 | 0.322 | 未崩且最优档：Score 0.347 与 v0 SFT 持平、F 0.033 为 InstrDialog 全表最低 → 换 seed 后 SFT+OPSD 在该流可稳定、非系统性失稳 |
| v1_SFT_OPSD | 加载 SFT 最终 adapter 初始化，继续 OPSD | InstrDialog++ | 0.300 | 0.313 | 0.084 | 0.082 | 0.409 | 0.402 | 0.457 | 0.412 | 0.000 | 0.200 | 0.272 | 合理：Score 0.300，F 0.084 显著优于 v0 的 0.180——SFT 初始化 + OPSD 在长流上收益明确 |
| v1_teacher_enhance_OPSD | SFT init OPSD + teacher 格式约束 | InstrDialog | 0.153 | 0.189 | 0.194 | 0.178 | 0.253 | 0.245 | 0.321 | 0.259 | 0.000 | 0.000 | 0.171 | 负向：Score 0.153 远低于 v0 的 0.346 与无约束 SFT+OPSD；格式约束在 19 段流上伤害大，NLG 同步掉 |
| v1_teacher_enhance_OPSD | SFT init OPSD + teacher 格式约束 | InstrDialog++ | 0.244 | 0.249 | 0.139 | 0.139 | 0.379 | 0.373 | 0.428 | 0.387 | 0.000 | 0.100 | 0.220 | 负向：Score 低于无约束 SFT+OPSD（0.300）与 v0（0.285）；F 0.139 介于二者之间，格式约束未兑现「压修饰词」的收益 |
| v1_seg_OPSD | 段内 SFT/OPSD 交替，K=25（约 1 epoch/phase） | InstrDialog | 0.289 | 0.304 | 0.156 | 0.156 | 0.376 | 0.372 | 0.415 | 0.371 | 0.000 | 0.000 | 0.280 | 网格最优：三条 K 中 Score 最高、F 最低；仍低于 v0 SFT（0.346 / 0.089） |
| v1_seg_OPSD | 段内 SFT/OPSD 交替，K=50（约 2 epoch/phase） | InstrDialog | 0.268 | 0.283 | 0.183 | 0.178 | 0.349 | 0.343 | 0.388 | 0.345 | 0.000 | 0.000 | 0.261 | 更长 phase 未带来收益，Score/F 均差于 K=25 |
| v1_seg_OPSD | 段内 SFT/OPSD 交替，K=75（约 3 epoch/phase） | InstrDialog | 0.205 | 0.247 | 0.232 | 0.199 | 0.285 | 0.278 | 0.343 | 0.279 | 0.100 | 0.100 | 0.227 | 最差：K 越大过拟合风险越高，Score 掉到 0.205 |
| v1_seg_OPSD | 胜出 K=25 打长流 | InstrDialog++ | 0.289 | 0.305 | 0.151 | 0.146 | 0.404 | 0.395 | 0.466 | 0.394 | 0.000 | 0.000 | 0.267 | 合理：Score 与 v0 持平（0.289 vs 0.285），F 0.151 优于 v0 的 0.180，但不如 SFT+OPSD 的 0.084 |
| v1_seg_OPSD_replay | 段内交替 K=25 + 20% replay | InstrDialog | 0.378 | 0.389 | 0.083 | 0.072 | 0.443 | 0.440 | 0.463 | 0.435 | 0.100 | 0.100 | 0.355 | 本轮最强短流：Score 0.378 超过 v0 的 0.346，F 0.083 与 v0 持平；20% replay 明显补上交替本身的遗忘 |
| v1_seg_OPSD_replay | 段内交替 K=25 + 20% replay | InstrDialog++ | 0.379 | 0.381 | 0.108 | 0.111 | 0.466 | 0.461 | 0.517 | 0.460 | 0.200 | 0.200 | 0.330 | 本轮最强长流：Score 0.379 超过 v0 的 0.285 与 SFT+OPSD 的 0.300；F 0.108 优于 v0 的 0.180；NLG（F1/ROUGE/BLEU）同为长流最高档 |
| v1_gold_OPSD | 同一步 CE + gold KL（λ=0.3） | InstrDialog | 0.341 | 0.357 | 0.106 | 0.089 | 0.403 | 0.400 | 0.430 | 0.399 | 0.000 | 0.000 | 0.327 | 合理：与 v0 SFT 几乎持平（0.341 vs 0.346），符合「主项仍是 CE」的设计；F 略差（0.106 vs 0.089） |
| v1_gold_OPSD | 同一步 CE + gold KL（λ=0.3） | InstrDialog++ | 0.303 | 0.316 | 0.157 | 0.147 | 0.402 | 0.397 | 0.464 | 0.394 | 0.100 | 0.100 | 0.276 | 合理：Score 0.303 略高于 v0 的 0.285、与 SFT+OPSD 的 0.300 接近；F 0.157 优于 v0 的 0.180，但弱于 SFT+OPSD 的 0.084 |
| v1_neg_OPSD | 模板负样本 pairwise + CE（μ=0.3） | InstrDialog | 0.325 | 0.336 | 0.100 | 0.089 | 0.383 | 0.378 | 0.424 | 0.377 | 0.000 | 0.000 | 0.308 | 合理偏弱：Score 0.325 略低于 v0 的 0.346，F 0.100 接近；格式 pairwise 未超过纯 SFT，NLG 略掉 |
| v1_neg_OPSD | 模板负样本 pairwise + CE（μ=0.3） | InstrDialog++ | 0.337 | 0.358 | 0.113 | 0.102 | 0.425 | 0.420 | 0.484 | 0.419 | 0.000 | 0.000 | 0.310 | 合理且较强：Score 0.337 超过 v0 的 0.285 与 SFT+OPSD 的 0.300，F 0.113 优于 v0 的 0.180；CNN/DM（seg 27）OOM 修复后整段跑通 |
| v1_teacher_enhance_OPSD | seed 123 重跑 | InstrDialog | 0.253 | 0.263 | 0.127 | 0.132 | 0.332 | 0.327 | 0.376 | 0.332 | 0.000 | 0.000 | 0.242 | 负向减弱：Score 0.253 高于原 0.153，仍低于 v0 与无约束 SFT+OPSD；原 0.153 偏悲观 |
| v1_teacher_enhance_OPSD | seed 456 | InstrDialog | 0.325 | 0.346 | 0.033 | 0.033 | 0.396 | 0.390 | 0.426 | 0.394 | 0.000 | 0.000 | 0.318 | 单次异常倾向：Score 0.325/F 0.033 接近无约束最优档，原 0.153 未复现，格式约束伤害非必然崩 |
| v1_teacher_enhance_OPSD | seed 123 重跑 | InstrDialog++ | 0.274 | 0.284 | 0.089 | 0.100 | 0.401 | 0.394 | 0.438 | 0.417 | 0.000 | 0.200 | 0.250 | 负向同向：Score 0.274 仍低于无约束 SFT+OPSD 原 0.300，长流格式约束未兑现收益 |
| v1_teacher_enhance_OPSD | seed 456 | InstrDialog++ | 0.263 | 0.284 | 0.108 | 0.100 | 0.401 | 0.394 | 0.444 | 0.410 | 0.000 | 0.200 | 0.250 | 负向同向：Score 0.263，双 seed 均低于无约束对照，长流结论稳 |
| v1_seg_OPSD_replay | seed 123 重跑 | InstrDialog | 0.357 | 0.362 | 0.072 | 0.078 | 0.419 | 0.416 | 0.454 | 0.411 | 0.100 | 0.100 | 0.332 | 合理且仍强：Score 0.357 超 v0 的 0.346，F 0.072 优于 v0；原 0.378 略乐观但方向同 |
| v1_seg_OPSD_replay | seed 456 | InstrDialog | 0.331 | 0.343 | 0.135 | 0.133 | 0.414 | 0.409 | 0.456 | 0.399 | 0.100 | 0.100 | 0.318 | 增益收窄：Score 0.331 低于 v0、F 0.135 变差 → 短流「最强」对 seed 敏感 |
| v1_seg_OPSD_replay | seed 123 重跑 | InstrDialog++ | 0.387 | 0.389 | 0.108 | 0.105 | 0.490 | 0.484 | 0.526 | 0.483 | 0.200 | 0.200 | 0.336 | 长流最强复现：Score 0.387 超原 0.379 与 v0/SFT+OPSD，NLG 仍最高档 |
| v1_seg_OPSD_replay | seed 456 | InstrDialog++ | 0.416 | 0.416 | 0.066 | 0.071 | 0.497 | 0.491 | 0.531 | 0.487 | 0.000 | 0.000 | 0.358 | 长流最强且更稳：Score 0.416/F 0.066，双 seed 同向确认 replay 长流收益 |
| v1_seg_OPSD | seed 123 重跑（K=25） | InstrDialog | 0.273 | 0.304 | 0.178 | 0.156 | 0.346 | 0.342 | 0.391 | 0.343 | 0.100 | 0.100 | 0.280 | 合理偏弱：Score 0.273 近原 0.289，仍低于 v0，网格胜出可复现但优势有限 |
| v1_seg_OPSD | seed 456（K=25） | InstrDialog | 0.005 | 0.093 | 0.421 | 0.340 | 0.126 | 0.122 | 0.187 | 0.153 | 0.000 | 0.000 | 0.085 | 异常崩盘：Score 0.005/F 0.421，K=25 短流对 seed 极度敏感；已另开 seed456 重跑+seed789 确认 |
| v1_seg_OPSD | seed 123 重跑（K=25） | InstrDialog++ | 0.316 | 0.339 | 0.159 | 0.135 | 0.417 | 0.409 | 0.471 | 0.409 | 0.100 | 0.100 | 0.293 | 合理：Score 0.316 略高于原 0.289，F 0.159 接近原 0.151 |
| v1_seg_OPSD | seed 456（K=25） | InstrDialog++ | 0.263 | 0.281 | 0.187 | 0.171 | 0.379 | 0.373 | 0.440 | 0.381 | 0.000 | 0.000 | 0.248 | 合理偏弱：Score 0.263 低于原跑，双 seed 未再现短流那种崩盘 |
| v1_SFT_OPSD | seed 123 重跑 | InstrDialog++ | 0.287 | 0.313 | 0.111 | 0.097 | 0.411 | 0.405 | 0.464 | 0.416 | 0.000 | 0.200 | 0.274 | 合理但原值偏乐观：Score 0.287 略低于原 0.300，F 0.111 仍优于 v0 的 0.180 |
| v1_SFT_OPSD | seed 456 | InstrDialog++ | 0.268 | 0.295 | 0.114 | 0.100 | 0.404 | 0.398 | 0.448 | 0.414 | 0.000 | 0.200 | 0.259 | 合理同向：Score 0.268/F 0.114，长流抗遗忘仍优于 v0，绝对值低于原单点 |
| v1_teacher_enhance_OPSD | seed123 rerun2 | InstrDialog | 0.258 | 0.274 | 0.094 | 0.094 | 0.342 | 0.336 | 0.377 | 0.341 | 0.000 | 0.000 | 0.246 | 负向减弱复现：Score 0.258 近首次重跑 0.253，高于原 0.153，仍低于 v0 与无约束 SFT+OPSD |
| v1_teacher_enhance_OPSD | seed789 | InstrDialog | 0.152 | 0.346 | 0.211 | 0.022 | 0.231 | 0.227 | 0.348 | 0.236 | 0.000 | 0.000 | 0.318 | 负向复现原档：Score 0.152 贴近原 0.153、F 0.211，格式约束伤害对 seed 高度敏感 |
| v1_seg_OPSD | seed456 rerun（K=25） | InstrDialog | 0.220 | 0.262 | 0.233 | 0.200 | 0.312 | 0.308 | 0.358 | 0.310 | 0.000 | 0.000 | 0.242 | 未崩：同 seed456 重跑 Score 0.220/F 0.233，未复现原 0.005 崩盘，属 GPU 非确定性 |
| v1_seg_OPSD | seed789（K=25） | InstrDialog | 0.325 | 0.341 | 0.111 | 0.106 | 0.394 | 0.390 | 0.432 | 0.391 | 0.000 | 0.000 | 0.313 | 未崩且合理：Score 0.325/F 0.111 近原 K=25 与 seed123，确认 0.005 崩盘非方法必然 |
