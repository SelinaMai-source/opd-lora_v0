# 科学审计（只读现有代码与产物）

审计对象是 `/root/autodl-tmp/rp_lora_v0` 的当前源码，以及 `results/runs` 里已有的 `final_metrics.json`。本次没有改源码、没有重训、没有重算评测。旧 run 仍属于它们当时的协议，不能改标成训练集留出监控。

`core/train.py` 在 `baseline_name == ce_opsd_replay` 时导入的是 `baselines/basic_baselines/ce_opsd_replay/method.py`。该文件与 `experiments_v3/methods/ce_opsd_replay/method.py` 当前内容一致。`metric_opsd_replay` 继承同一个类。下面的行号按这次阅读时的文件。

## 问题

### AUD-P0-001 用官方 eval 的正确率选择退化回放

- 严重级别：P0
- 位置：`baselines/basic_baselines/ce_opsd_replay/method.py`（与 `experiments_v3/methods/ce_opsd_replay/method.py` 相同）`_select_forgotten`（约 329–346 行）、`on_eval_end`（约 93–104 行）；调用点 `core/train.py` 约 552–570 行；遗忘向量来自 `core/evaluate.py` 约 278–294 行，评测样本来自 `segment.eval`（`_eval_segment` 约 599 行）。
- 描述：每段训练结束后，`evaluate_stream` 只在已经见过的段上、用每段官方 eval（两个 benchmark 文件都是每段 10 条，且与 train 的 `(instruction, input)` 重叠为 0）计算 strict accuracy 的遗忘。`on_eval_end` 把 `extra.forgetting_by_segment` 写入方法对象。下一段若 `strategy=proto` 且 `forgotten_ratio>0`，`_select_forgotten` 取遗忘最大的一个已见段，再从该段的训练缓冲里抽样。遗忘不大于 0 时退回全体记忆的均匀抽样，但这些样本仍记在 forgotten 槽。它不读取尚未训练的段。被回放的文本来自训练缓冲，选择哪一段用的是测试集正确率。
- 受影响实验：快照里 `baseline_name` 为 `ce_opsd_replay` 或 `metric_opsd_replay`、`replay.strategy=proto` 且 `forgotten_ratio>0` 的 run。清单里标了 `eval_split_monitor` 的有 35 个，其中全流程 18 个：`v3_proto_ce` 两个数据集、`v3_proto_opsd` 的 seed 456/789/42 两个数据集、`v3_proto_coverage` 两个数据集、`v3_proto_metric_gate_opsd` 的 μ=0/0.02/0.05 两个数据集、`v3_transfer_all_forgotten` 两个数据集。另有 15 个单段 isolation 和 2 个两段冒烟。`strategy=uniform` 的 `v3_rand_*` 不调用 `_select_forgotten`，即使 yaml 里仍写着 `forgotten_ratio: 0.1`。`neighbor_ratio: 0` 的全近邻 run 不走这条路径。
- 产物核对：`save_segment_adapters: true` 的 `v3_proto_opsd` 六个 run 和 `v3_proto_coverage` 两个 run 有 `segment_dumps`。把第 t 段 `replay_log` 的 forgotten 来源与第 t-1 段 dump 的 `forgetting_by_segment` 对照，来源要么就是 argmax，要么是 `best_f<=0` 的均匀退回；没有第三种不匹配。以 `v3_proto_opsd_instrdialog_seed456` 为例，18 个有 forgotten 样本的段里 15 个命中 argmax、3 个是非正遗忘退回。`v3_proto_ce`、metric gate 和 `v3_transfer_all_forgotten` 的 `save_segment_adapters` 为 false，没有 dump，不能做同样的连接；它们的配置和方法类与上述路径相同。
- 必须的修正：退化段只在训练样本的留出集上估计，eval 不参与选样。旧目录保持原协议，不改写成 train-holdout 合法结果。
- 验证：读上述函数；对有 dump 的 run 做段级连接。本次没有改代码。
- 处理状态：未修。源码树按本次任务保持不动。

### AUD-P1-001 默认近邻槽不是按余弦取最近段

- 严重级别：P1
- 位置：`baselines/basic_baselines/ce_opsd_replay/method.py` `_select_neighbors`（约 294–311 行）、`_sample_examples`（约 564–569 行）。类注释写的是 nearest-compatible；函数注释写的是 format-compatible 之后均匀抽样，余弦只给池子排序。
- 描述：默认 `transfer_slot=neighbor` 先按 `task_score_type` 和 gold 长度比过滤历史段，再把这些段的 K 条缓冲全部放进一个池，`random.sample` 抽取。排序不改变无放回均匀抽样。因此原型向量没有决定近邻槽抽到哪些样本。`get_activations_tensor` 对池化隐状态做了 L2 归一化，所以 `_cosine` 里的点积在数值上等于余弦，但这条分数在默认近邻槽里不被用来抽样。`transfer_slot=anti` 会取余弦最低的一个兼容段，那条路径确实使用该分数。
- 受影响实验：所有默认 neighbor 槽的 proto run，包括 12 个已完成因子格里的 PGDR 格，以及 coverage、metric gate、全近邻。不能把这些结果写成“按原型余弦检索最近邻样本”。
- 必须的修正：若论文声称最近邻检索，应改为按余弦选择段或样本；否则正文应写成格式兼容后的均匀抽样。旧数字不要改口。
- 验证：阅读上述函数。未改代码。
- 处理状态：未修。

### AUD-P1-002 九个 Lit 冒烟目录里的 final_metrics 不是测量

- 严重级别：P1
- 位置：`results/runs/lit_{clora,gorp,inscl,kpig,onedit,saber,sapt,ssr,treelora}_instrdialog_seed456_smoke2seg/final_metrics.json`
- 描述：这 9 个文件只有 `eval.seen_avg_score` 和 `eval.forgetting`，没有其余 9 个指标，没有 `config_snapshot.yaml`、`run_manifest.json` 或 `eval_debug`。它们不能进入任何对照表。`lit_gainlora` 与 `lit_sdft` 的同名冒烟目录有完整指标和配置，是两段实测，清单里标成冒烟，不与全流程混比。
- 受影响实验：仅这 9 个占位目录。
- 必须的修正：不要引用这些数字。真正的冒烟需要完整 `final.eval` 与配置快照。
- 验证：直接读 JSON 键。
- 处理状态：未删除源目录。清单中 `verification_status=placeholder_not_a_measurement`，表已排除。

### AUD-P1-003 CITB / LCIA 与 v3 不是同一个训练行预算

- 严重级别：P1
- 位置：这些 run 的 `config_snapshot.yaml` 使用 `citb_cl_dialogue_tasks_train50_eval10.json` 或 `citb_cl_38_random_tasks_train50_eval10.json`，没有 `*_multiref.json`。`sequential_lora` 的 `train_on_segment` 直接把 `segment.train` 的每一行送进 CE。
- 描述：两个非 multiref 文件的唯一 `(instruction, input)` 仍是每段 50 条 train、10 条 eval，eval 与 train 无键重叠。但同一输入的多条 gold 被拆成多行。InstrDialog train 行数 1012，对唯一输入 950；InstrDialog++ 为 2241 对 1900。multiref 文件把多 gold 放在同一行的 `outputs` 里，行数就是 950 和 1900。实测 optimizer step：`v2pre_citb_sequential_instrdialog_seed456` 为 507，`...instrdialogpp...` 为 1122；同模型上走 multiref、无 replay 的 Lit 全流程（例如 CLoRA）为 475 和 950。v3 因子格因为 20% replay 是 565 和 1135。另外这 36 个 CITB/LCIA 快照没有 `eval_normalization.format_normalize` 键。当前 `normalize_for_eval` 在缺键时默认开启格式归一，但本次没有重放评测，不能把“当前默认”写成“当时日志外又证明了一次”。表中的 CITB 数字只取各 run 自己的 `final.eval`。`v2pre_citb_sequential_instrdialog_seed456` 的 Acc 0.300、F 0.139 与已入表参考一致，协议差异仍然存在。
- 受影响实验：33 个带 `final_metrics.json` 的 `v2pre_citb_*` 和 3 个 `v2pre_lcia_*`（含冒烟）。
- 必须的修正：和 v3 对照时单独标训练行数与流文件，不写成匹配预算。
- 验证：统计两个流文件的行数与唯一键；核对 `train.batches` 之和。
- 处理状态：未重跑。CITB/LCIA 放在 `results/tables/eleven_metrics_citb_lcia.csv`，不放进 v3/Lit 的同协议表。

### AUD-P2-001 近邻槽和退化槽可以重复同一条样本

- 严重级别：P2
- 位置：`_mix_current_and_replay` 把两个槽拼接，中间没有按 `(instruction, input)` 去重。
- 描述：`replay_log.jsonl` 的 `unique_pairs < n_replay` 时，该段回放里有重复对。去重后的完成 pass 上，`v3_proto_opsd_instrdialog_seed456` 有 6 段、共 10 个重复槽位；`v3_proto_opsd_instrdialogpp_seed456` 同样是 6 段、10 个槽位。四个 `v3_rand_*` 的均匀单槽为 0。
- 受影响实验：proto 双槽 run。配额名义上是 10%+10%，有效唯一条数可以更少。
- 必须的修正：若配额指唯一样本，跨槽去重并补足。旧日志保持原样。
- 验证：对 `replay_log.jsonl` 做 `n_replay - unique_pairs`。重复段的段号见 `results/tables/replay_selector_audit.csv`。
- 处理状态：未修。

### AUD-P2-002 forgotten 计数包含均匀退回

- 严重级别：P2
- 位置：`_select_forgotten` 在 `best_f<=0` 时仍把样本标成 `forgotten`。
- 描述：段计数 `n_forgotten` 不等于“来自正遗忘段的样本数”。有 dump 的 proto run 里，退回段数见 `replay_selector_audit.csv` 的 `forgotten_nonpositive_fallback`。
- 受影响实验：与 AUD-P0-001 相同、且发生过非正遗忘的段。
- 必须的修正：日志把退回标成 uniform fallback。
- 验证：dump 连接。
- 处理状态：未修。

### AUD-P2-003 一条随机 CE 日志里有早段前缀

- 严重级别：P2
- 位置：`results/runs/v3_rand_ce_instrdialog_seed456/metrics.jsonl` 有 29 行，`replay_log.jsonl` 有 30 行；段 0–9 各出现两次。`logs/v3_rand_ce_instrdialog_seed456.log` 只记录了一次段 0–18。
- 描述：`final_metrics.json` 的末段是 segment 18、`num_seen_segments=19`，对应日志里的那一次完整 pass。前缀里的 CE 和 seen-avg 与后一次不同，所以不是同一行写了两遍。本次 step 与回放合计只用每个 `segment_id` 的最后一条，InstrDialog 随机 CE 为 565 step、180 条回放，与另外三个 InstrDialog 因子格的 step 数相同。不能把 29 行相加当成这次训练的 step。
- 受影响实验：该 run 的过程日志。末点 11 个指标仍来自最后一段。
- 必须的修正：引用过程量时丢掉前缀，或在日志里分开 run id。
- 验证：段 id 计数，以及 train log 的段头。
- 处理状态：清单已加 `metrics_jsonl_repeated_segment_rows=10`。未改原日志。

### AUD-P2-004 记录的 git SHA 来自脏工作区

- 严重级别：P2
- 位置：各 run 的 `run_manifest.json`：`git_commit_sha` 与 `git_status_porcelain`。
- 描述：已核对的 v3 因子格 SHA 都是 `32225c8af3de60c467a77fef63c77bab996c1e07`，同时 porcelain 有数百条。`v3_proto_opsd_instrdialog_seed456` 的清单在启动时把 `baselines/basic_baselines/ce_opsd_replay/method.py` 和 `core/evaluate.py` 标成已修改。SHA 不是这些 run 的干净代码快照。CITB sequential 的 SHA 是另一个提交 `bcdeaaa2a30bdf8f1736ab774c1f513537bb3629`，同样带 porcelain。
- 受影响实验：所有带 `run_manifest.json` 且 porcelain 非空的 run。
- 必须的修正：可复现包要保存实际执行的方法文件，而不是只保存脏树上的 HEAD。
- 验证：读 manifest。本次没有执行 git 写操作。
- 处理状态：未改历史 manifest。

### AUD-P2-005 三个 baseline 的评测使用 oracle 段号

- 严重级别：P2
- 位置：`baselines/citb/adaptercl/method.py` 文档字符串写明 eval 按 oracle `source_segment_id` 切适配器；`baselines/treelora/method.py` 与 `baselines/saber/method.py` 调用 `install_oracle_eval_hook`。
- 描述：这三路评测看得到当前样本所属段。PGDR/OPSD 的评测提示是 `format_for_infer` 的 instruction+input，不切段适配器。它们可以同表展示，但不能写成无任务号的匹配比较。
- 受影响实验：对应的 `v2pre_citb_adaptercl*`、`lit_treelora*`、`lit_saber*` 全流程与冒烟。
- 必须的修正：表上标明 oracle task id。
- 验证：读这三个方法文件。没有把其余 Lit 方法推断成 oracle。
- 处理状态：未改实现。

### AUD-P2-006 CITB replay 的 yaml 比例不是实现用的预算

- 严重级别：P2
- 位置：`baselines/citb/replay/method.py` 使用 `replay_k()`；`baselines/citb/common.py` 的 `replay_k` 读取 `replay.k` 或 `citb.k`，不读 `replay.replay_ratio`。
- 描述：`v2pre_citb_replay10_instrdialog_seed456` 的快照同时有 `replay.k: 10` 和 `replay.replay_ratio: 0.3`。实现用的是 K=10。清单里的 `replay_ratio` 列保留快照原值 0.3，`memory_capacity` 为 10。
- 受影响实验：带这两套键的 CITB replay 配置。
- 必须的修正：引用预算时用 K，不把 0.3 当成回放比例。
- 验证：读 `replay_k` 与该快照。
- 处理状态：未改配置。

## 已核对、不单列成缺陷的实现

### PGDR 记忆

- 每段从该段 train 里按输入轮转抽出最多 `buffer_k` 条（因子格快照为 10），在该段训练结束后追加。当前段的回放池不含本段，也不含未来段。
- 段向量是冻结底座（`disable_adapter`）对指令文本的池化、L2 归一化表示。
- 训练提示不含任务号。段号只用于记忆和遗忘表。
- 因子格快照：`neighbor_ratio` 与 `forgotten_ratio` 在 proto 上各 0.1，均匀策略只用 `replay_ratio: 0.2`。`teacher_proto.mode` 为 `none`。

### OPSD

位置：`baselines/basic_baselines/opsd/method.py` 的 `run_opsd_step`、`token_divergence`、`_teacher_forward_logits`；组合损失在 `baselines/basic_baselines/ce_opsd/method.py`。

- 学生续写来自当前策略：`sample_rollouts`，因子格快照 `rollout_temperature: 1.0`、`rollout_top_p: 1.0`。空续写跳过。
- 教师是冻结底座：`torch.no_grad()` 加上 PEFT `disable_adapter()`。学生 logits 在这个 `no_grad` 之外计算，组合损失一次 `backward`，`run_opsd_step(..., do_step=False)` 避免第二次更新。
- 教师提示不是学生提示。`format_for_teacher` 始终放入 `<privileged_reference_output>`。`teacher_format_constraint: true` 时再追加 “Preserve its key content, answer format, wording style, and approximate length.” 十二个因子格快照都是 true。参考句是训练 gold 中与续写解码 Token F1 最高的一条，ROUGE-L 再决胜，不是 eval gold。
- 散度是广义 JSD，β=0.5：`m = β p_T + (1-β) p_S`，`β KL(p_T‖m) + (1-β) KL(p_S‖m)`，对续写 token 取均值，再对样本取均值。`log_softmax` 没有单独的蒸馏温度。本次不做温度扫描。
- `lambda_opsd<=0` 时不采样、不回传 OPSD。因子格里两个 CE 格的 λ 是 0。λ>0 时损失是 CE + λ·JSD。回放样本和当前样本在同一个混合列表里，因此都进 CE；序列长于等于 1024 的样本跳过 OPSD。这十二个 run 的末段日志 `opsd_skipped_long` 为 0。
- 日志里没有教师熵、学生熵、token 级 KL 或 CE/OPSD 梯度余弦。没有这些字段就不能报告它们。

### 十一指标

`evaluate_stream` 的末点写入 `final_metrics.json` 的 `final.eval.*`。生成配置在 `eval_debug` 里是 `do_sample: false`。

| 指标 | 键 | 实现 |
|---|---|---|
| Seen-Avg Acc | `seen_avg_score` | 各已见段 strict EM 的算术平均 |
| Seen-Avg Task-aware Acc | `seen_avg_task_aware_score` | 各段 task-aware EM 的算术平均 |
| Forgetting | `forgetting` | 只对非当前段，`max(0, 历史峰值 − 当前)` 的平均；当前段记 0 且不进平均 |
| Task-Aware Forgetting | `task_aware_forgetting` | 同上，用 task-aware 准确率 |
| Token F1 / ROUGE-L / BLEU / LCS | `token_f1_mean` 等 | 每条样本对全部 gold 取最佳对后，再对样本平均 |
| Current-Seg Acc | `current_score` | 最后一个已见段的 strict EM |
| Current-Seg Task-aware Acc | `current_task_aware_score` | 该段 task-aware EM |
| Task-aware Score Mean | `task_aware_score_mean` | 样本级 task-aware 分数的平均 |

多 gold：任一 gold EM=1 优先，否则取 Token F1 最大的一对；task-aware、ROUGE-L、BLEU、LCS 跟同一对。`eval_normalization.format_normalize: true` 的 run 走对称格式归一。v3 与全流程 Lit 的快照有这个键。

这两个 benchmark 的 eval 每段都是 10 条，所以段宏平均和样本微平均在 strict Acc、task-aware Acc 上重合。十二个因子格里 `seen_avg_task_aware_score` 与 `task_aware_score_mean` 的差是 0。这不是两个独立估计。Forgetting 仍是原始下降量，越小越好；符号翻转只用于“越大越好”的交互对照，不替换原值。

Task-aware 与 strict 不是同一个判定：短标签、yes/no、数字和选项会抽取后再比。样本可以 strict 失败而 task-aware 成功。见轨迹文档里的实测行。

## 暂存仓库后续（不改源码树结论）

上面的 P0 仍描述 `/root/autodl-tmp/rp_lora_v0`：那棵树没有改，旧 `v3_*` 数字仍是 `eval_split_monitor`。本仓库 `methods/pgdr.py` 同时保留该选择器，并增加 `train_holdout`。`v3h_*` seed 456 的 8 个全流程 run 用训练集留出 5/50，`on_eval_end` 不把官方 eval 遗忘写入选择器。这 8 个 run 以及旧协议补跑的 12 个 seed 789/42 run（`v3_rand_ce`、`v3_proto_ce`、`v3_rand_opsd` × 两个数据集）的 `final_metrics.json` 在 `/root/autodl-tmp/rp_lora_v3_runs/runs/`。已有的 `v3_proto_opsd` seed 456/789/42 没有重跑，清单里的数字仍指向源码树产物。冒烟 `v3h_proto_opsd_instrdialog_seed456_smoke2seg` 只有 2 段，不与全流程混比。
