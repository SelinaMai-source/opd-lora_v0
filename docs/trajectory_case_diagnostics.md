# 已有轨迹、回放计数与实测案例

来源只包括 seed 456 的四个因子格、`v3_proto_opsd` 的 seed 789/42，以及 `v2pre_citb_sequential_*_seed456` 的 `segment_*/eval_metrics.json`、`metrics.jsonl`、`replay_log.jsonl` 和最终 `eval_debug`。没有温度扫描。十二个因子格快照的 `rollout_temperature` 都是 1.0；`token_divergence` 的 `log_softmax` 没有第二个温度。

过程日志里有的诊断是每段平均 CE、OPSD（λ>0 时）、rollout 平均 token 数和 grad norm。没有教师熵、学生熵、逐 token KL，也没有 CE 与 OPSD 的梯度余弦。下表是各段最后一条日志的平均，InstrDialog 随机 CE 已去掉段 0–9 的前缀。

| run | 段数 | 平均 CE | 平均 OPSD | 平均 rollout token | 平均 grad norm |
|---|---:|---:|---:|---:|---:|
| `v3_rand_ce_instrdialog_seed456` | 19 | 1.3222 | — | — | 6.303 |
| `v3_proto_ce_instrdialog_seed456` | 19 | 1.2423 | — | — | 6.315 |
| `v3_rand_opsd_instrdialog_seed456` | 19 | 1.3139 | 0.1562 | 9.610 | 6.535 |
| `v3_proto_opsd_instrdialog_seed456` | 19 | 1.2265 | 0.1501 | 9.627 | 5.665 |
| `v3_rand_ce_instrdialogpp_seed456` | 38 | 1.1199 | — | — | 5.346 |
| `v3_proto_ce_instrdialogpp_seed456` | 38 | 1.0363 | — | — | 5.738 |
| `v3_rand_opsd_instrdialogpp_seed456` | 38 | 1.1171 | 0.1366 | 9.654 | 5.714 |
| `v3_proto_opsd_instrdialogpp_seed456` | 38 | 1.0394 | 0.1360 | 9.154 | 5.274 |

λ=0 的 CE 格没有 OPSD 应用记录。OPSD 格的段平均 JSD 大约 0.13–0.16，续写大约 9–10 个 token。这只说明损失的量级，不能单独解释 InstrDialog++ 上 OPSD 变差。

## 回放计数

去重到每个段最后一条 `replay_log` 之后：

- 四个随机格：forgotten 与 neighbor 都是 0，全部落在 uniform。InstrDialog 合计 180 条，InstrDialog++ 合计 370 条。段 0 没有历史记忆，所以不是 19×10 或 38×10。
- 六个 `v3_proto_opsd` 与两个 `v3_proto_ce`：neighbor 与 forgotten 条数相等。InstrDialog 各 90，InstrDialog++ 各 185。
- `v3_transfer_all_neighbor_*`：只有 neighbor。`v3_transfer_all_forgotten_*`：只有 forgotten。这两对没有 segment dump，不能把 forgotten 条数再分成 argmax 与退回。

有 dump 的 proto OPSD / coverage run 上，forgotten 来源与上一段 eval 遗忘的 argmax 一致，或者是非正遗忘时的均匀退回。详见审计 AUD-P0-001 和 `results/tables/replay_selector_audit.csv`。

## 轨迹分类

矩阵来自每段 `eval_metrics.json` 的 `per_segment_accuracy`。任务 i 的序列只保留 stage ≥ i。分类用 strict EM，规则是互斥的：

- `never_learned`：序列最大值是 0。
- `retained`：最大值大于 0，且之后没有掉到当时运行最大值以下。
- `forgotten_then_recovered`：出现过下降，末点又回到历史峰值。
- `learned_then_forgotten_with_partial_recovery`：末点低于峰值，但下降之后有过回升。
- `learned_then_forgotten`：末点低于峰值，且没有这种回升。

`v3_proto_opsd_instrdialog_seed456` 的五个例子：

- 任务 0：全程 0，never learned。
- 任务 5：从首次评测起一直是 0.700，retained。
- 任务 3：从 1.000 降到 0.500 后再回到 1.000，forgotten then recovered。
- 任务 1：峰值 0.200、末点 0.100，中间有回升，partial recovery。
- 任务 10：峰值 0.100、末点 0，learned then forgotten。

seed 456 的任务计数：

| run | never | retained | recovered | partial | forgotten |
|---|---:|---:|---:|---:|---:|
| Sequential InstrDialog | 6 | 2 | 2 | 3 | 6 |
| Random+CE | 6 | 3 | 3 | 3 | 4 |
| PGDR+CE | 8 | 3 | 2 | 4 | 2 |
| Random+OPSD | 6 | 3 | 4 | 2 | 4 |
| PGDR+OPSD | 6 | 5 | 5 | 2 | 1 |
| Sequential InstrDialog++ | 9 | 2 | 4 | 17 | 6 |
| Random+CE | 10 | 2 | 6 | 13 | 7 |
| PGDR+CE | 12 | 5 | 9 | 10 | 2 |
| Random+OPSD | 9 | 5 | 6 | 13 | 5 |
| PGDR+OPSD | 11 | 4 | 6 | 10 | 7 |

InstrDialog 上，PGDR+OPSD 的“末点仍低于峰值且没有回升”只有 1 个任务，随机 CE 是 4 个。InstrDialog++ 上这个计数是 7，高于 PGDR+CE 的 2。这和末点遗忘的方向一致：InstrDialog++ 的 PGDR+CE 遗忘 0.035，PGDR+OPSD 是 0.068。计数依赖上面的阈值，不是另一套官方指标。完整序列在 `results/trajectories/task_classes.csv`。

## seed 456 交互

用未四舍五入的末点。Forgetting 的原始交互越小表示 PGDR 与 OPSD 的组合比相加主效应更降低遗忘；表中同时给出取负后的“越大越好”值。

| benchmark | 指标 | PGDR 主效应 | OPSD 主效应 | 原始交互 | 越大越好的交互 |
|---|---|---:|---:|---:|---:|
| InstrDialog | Seen-Avg Acc | 0.010526 | 0.026316 | 0.042105 | 0.042105 |
| InstrDialog | Forgetting | −0.025000 | −0.030556 | −0.005556 | 0.005556 |
| InstrDialog++ | Seen-Avg Acc | 0.078947 | −0.005263 | −0.036842 | −0.036842 |
| InstrDialog++ | Forgetting | −0.071622 | 0.009459 | 0.045946 | −0.045946 |

InstrDialog 上，OPSD 主效应提高 Acc、降低遗忘，交互也为正 Acc。InstrDialog++ 上，PGDR 主效应提高 Acc、降低遗忘；OPSD 主效应略降 Acc、略增遗忘；Acc 交互为负。只有一个 seed，不能把交互写成稳定的统计结论。其余 9 个指标在 `results/tables/factorial_interaction_seed456.csv`。

## 实测案例

案例来自 seed 456 最终 `eval_debug/eval_segment_*.json` 的最后一段，checkpoint 是各 run 的 `final_adapter`。同一条用 `(source_segment_id, source_example_idx)` 对齐，并核对了指令一致。下面每组只引用该组里最短的一条；候选数量是严格 EM 条件成立的条数，不是抽样估计。

### InstrDialog，段 3 任务 `task1730_personachat_choose_next`，样本 1

指令要求从四个候选里选出对话续写。Gold 是 “The keys I've done piano since I was 5.” 归一化后为 `the keys i've done piano since i was 5`。两个 PGDR run 预测这句，strict 与 task-aware 都对，Token F1、ROUGE-L、BLEU、LCS 都是 1。两个随机 run 预测 “I like instrumental; so many lyrics are so hard to understand.”，四项分别为 0.100、0.100、0.114、0.111。同指令下 PGDR 相对随机回放多对了这一条；OPSD 没有改变这四个预测的对错。满足该对照的 eval 条数是 8。

### InstrDialog，段 7 任务 `task1603_smcalflow_sentence_generation`，样本 1

输入 “What day of the week is tomorrow?” Gold “Tomorrow is Wednesday.” 两个 CE run 预测 “Tomorrow is Thursday.”，Token F1 0.667。两个 OPSD run 预测 gold，四项都是 1。这是随机回放下 OPSD 相对 CE 的严格 EM 差异，候选 7 条。它不能说明 OPSD 在所有任务上都更好。

### InstrDialog，段 6 任务 `task1500_dstc3_classification`，样本 7

Gold “Don't Care”。随机 CE、PGDR+CE、随机+OPSD 都预测 “Don't Care”。PGDR+OPSD 预测 “Expensive”，strict 与 task-aware 都失败，Token F1 与 ROUGE-L 为 0，BLEU 0.184，LCS 0。这是该数据集上 PGDR 条件下、旧段 strict EM 从有 OPSD 变为无的唯一一条。

### InstrDialog，段 7 同 smcalflow 任务，样本 3

Gold “Yes.” PGDR+OPSD 预测 “Yes, they do.” 归一化预测 `yes, they do`。strict EM 为假，task-aware 为真，Token F1、ROUGE-L、LCS 为 0，BLEU 0.250。随机+OPSD 预测 “Yes, Walk-ons in Mobile has a full bar.”，同样是 strict 假、task-aware 真。随机 CE 预测 “Yes”，两项都真。这类分歧在 PGDR+OPSD 的最终评测里有 3 条。

### InstrDialog++，段 33 任务 `task1384_deal_or_no_dialog_classification`，样本 8

Gold “No”。四个方法都预测 “No”，八个生成指标都是 1。这是旧任务上的共同保留。PGDR+OPSD strict 正确的旧段样本有 170 条，其中这一条四个方法都对。

### InstrDialog++，段 11 任务 `task1360_numer_sense_multiple_choice_qa_generation`，样本 2

Gold “a”。PGDR+CE 预测 “a”。另外三个方法预测 “d”。PGDR+OPSD 相对 PGDR+CE 在这条旧样本上失败。满足该条件的条数是 19。这与末点 Acc 从 0.474 降到 0.450 的方向一致，但一条预测不能解释全部 0.024 的差。

### InstrDialog++，段 37 当前段 `task1603_smcalflow_sentence_generation`，样本 1

Gold 仍是 “Tomorrow is Wednesday.” 两个 OPSD run 预测正确。PGDR+CE 预测 “Tomorrow is a Tuesday.”，随机 CE 预测 “Tomorrow is a Friday.”，Token F1 都是 0.571。PGDR+OPSD 在当前段 strict 正确的样本只有这 1 条。InstrDialog 的四个 seed 456 格在最后一段的 Current-Seg Acc 都是 0，所以没有“当前段 PGDR+OPSD 严格正确”的案例。

观察：同一 seed 上，PGDR 与 OPSD 都会在部分旧样本上改变 strict EM，方向随数据集和槽位变化。InstrDialog++ 的末点显示 OPSD 加在 PGDR 上之后 Acc 下降、遗忘上升。这是一个 seed 的末点和案例计数，不是因果证明。
