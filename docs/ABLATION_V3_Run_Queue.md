# ABLATION v3 Run Queue

单卡顺序。已有 `final_metrics.json` 且配置指纹一致则跳过。目录已存在但不完整则拒绝覆盖，续跑命令在 `/root/autodl-tmp/rp_lora_v3_runs/logs/<run>.resume_cmd`。

结果目录：`/root/autodl-tmp/rp_lora_v3_runs/runs`。

## 已完成矩阵（不重跑）

- 冒烟 `v3h_proto_opsd_instrdialog_seed456_smoke2seg`
- 旧协议 12（seed 789/42 的 rand_ce / proto_ce / rand_opsd × 两流）
- v3h 8（seed 456 train_holdout 2×2×两流）
- 历史 20% 随机 / 全相关 / 全退化 / 10+10、CITB / LCIA / v0 / v1 / v2 / Lit

## Pending extensions 顺序

E2 → E4 λ → E4 off-policy → E6 → E3 → E9 → E10 → E11。`forgotten_ratio>0` 的新 run 用 `train_holdout`，run_name 带 `v3h_`。

### E2 15/5 与 5/15 槽混合（train_holdout）
- `v3h_proto_opsd_n15f5_instrdialog_seed456`: done
- `v3h_proto_opsd_n5f15_instrdialog_seed456`: done
- `v3h_proto_opsd_n15f5_instrdialogpp_seed456`: done
- `v3h_proto_opsd_n5f15_instrdialogpp_seed456`: done

### E4 λ ∈ {0.05, 0.10, 0.50}（PGDR+CE+OPSD holdout）
- `v3h_proto_opsd_lam005_instrdialog_seed456`: done
- `v3h_proto_opsd_lam010_instrdialog_seed456`: done
- `v3h_proto_opsd_lam050_instrdialog_seed456`: done
- `v3h_proto_opsd_lam005_instrdialogpp_seed456`: done
- `v3h_proto_opsd_lam010_instrdialogpp_seed456`: done
- `v3h_proto_opsd_lam050_instrdialogpp_seed456`: done

### E4 off-policy gold tokens vs on-policy（λ=0.2 holdout）
- `v3h_proto_opsd_offpol_instrdialog_seed456`: done
- `v3h_proto_opsd_offpol_instrdialogpp_seed456`: done

### E6 replay_ratio ∈ {0, 0.10, 0.30, 0.40}
- `v3_proto_opsd_rr0_instrdialog_seed456`: done
- `v3h_proto_opsd_rr10_instrdialog_seed456`: done
- `v3h_proto_opsd_rr30_instrdialog_seed456`: done
- `v3h_proto_opsd_rr40_instrdialog_seed456`: done
- `v3_proto_opsd_rr0_instrdialogpp_seed456`: done
- `v3h_proto_opsd_rr10_instrdialogpp_seed456`: done
- `v3h_proto_opsd_rr30_instrdialogpp_seed456`: done
- `v3h_proto_opsd_rr40_instrdialogpp_seed456`: done

### E3 v3 协议对照（direct OPSD / seg_opsd K=25）
- `v3_opsd_direct_instrdialog_seed456`: done
- `v3_opsd_direct_instrdialogpp_seed456`: done
- `v3_seg_opsd_k25_instrdialog_seed456`: done
- `v3_seg_opsd_k25_instrdialogpp_seed456`: done

### E9 buffer_k ∈ {5, 20, 50}
- `v3h_proto_opsd_bufk5_instrdialog_seed456`: done
- `v3h_proto_opsd_bufk20_instrdialog_seed456`: done
- `v3h_proto_opsd_bufk50_instrdialog_seed456`: done
- `v3h_proto_opsd_bufk5_instrdialogpp_seed456`: done
- `v3h_proto_opsd_bufk20_instrdialogpp_seed456`: done
- `v3h_proto_opsd_bufk50_instrdialogpp_seed456`: done

### E10 InstrDialog 任务顺序置换
- `v3h_proto_opsd_ordrev_instrdialog_seed456`: done
- `v3h_proto_opsd_ordshuf_instrdialog_seed456`: done

### E11 holdout 全相关 / 全退化 20%
- `v3h_proto_opsd_allnb_instrdialog_seed456`: done
- `v3h_proto_opsd_allfg_instrdialog_seed456`: done
- `v3h_proto_opsd_allnb_instrdialogpp_seed456`: done
- `v3h_proto_opsd_allfg_instrdialogpp_seed456`: done

### Blocked

- E3 SFT→OPSD：无合法 init adapter（Sequential 快照为非 `*_multiref.json` 且无 `format_normalize`）。
- 蒸馏温度扫描：不做。原目标 JSD 无单独 T，`rollout_temperature` 为 1.0。

## 不做

不重跑 CITB / LCIA / v0 / v1 / v2 / Lit。不开 50–90% replay。不开新的 seed 789/42。
