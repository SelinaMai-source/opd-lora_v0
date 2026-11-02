# P0 监控审计（Protocol B）

审计仓库是 `/root/research-repos/opd-lora_v3`，分支 `v3`，HEAD `0f57c1c`。只读代码、multiref 行数和已有 run 的 `metrics.jsonl` / `config_snapshot.yaml`。没有启动 GPU 训练，没有新写 16 个 run 的 yaml，没有改历史 run，没有改配额函数或近邻选择器。

## 结论

`monitor: train_holdout` 路径上，官方测试集的遗忘**没有**进入回放选样。`CeOpsdReplayMethod.on_eval_end` 在该模式下直接返回。退化槽用的是训练集留出 5 条的 strict accuracy。8 个已完成的 `v3h_*_seed456` 全流程日志与这一路径一致。

训练条数已经改变：每个任务从 Protocol A 的 50 条训练变成 fit 45 + 监控 5。这是混杂因素，不能把 Protocol A/B 的差值全部写成“消泄漏”。

允许开始 16 个新 run。新 yaml 必须显式写 `monitor: train_holdout` 和 `holdout_seed: 456`。`scripts/write_v3_configs.py` 在 `train_holdout` 时把 `holdout_seed` 写成训练 seed（约第 120–122 行），不能原样用来生成这 16 个配置。本次没有生成这些 yaml。

仍要单独写明、且本次不改代码的限制：uniform 回放 9 条、proto 回放 8 条；近邻槽是格式兼容后的均匀抽样，不是余弦检索；fit 从 50 降到 45。另外，官方 `seen_avg_score` 仍可能触发默认硬停止，但它不写选择器；8 个全流程都跑完了，本次不关这个默认。

## 十三条泄漏问题

| Audit Item | Implementation Location | Existing Behavior | Risk | Evidence | Resolution |
|---|---|---|---|---|---|
| 1. 历史退化分数用哪份数据计算 | `methods/pgdr.py` `score_holdout_map`、`forgetting_from_accuracies`；调用点 `baselines/basic_baselines/ce_opsd_replay/method.py` `_refresh_holdout_forgetting`。Protocol A 则是同文件 `on_eval_end` 读 `core/evaluate.py` `evaluate_stream` 写入的 `extra.forgetting_by_segment` | `train_holdout`：每段训练结束后，对已保存的训练留出集做 strict accuracy，再按 `max(0, 历史峰值 - 当前)` 得到遗忘，当前段贡献 0。`eval_split_monitor`：把官方 eval 的遗忘写入 `_forgetting_by_segment` | 只有漏写 `monitor: train_holdout` 时才会回到官方 eval。类默认值就是 `eval_split_monitor`（`CeOpsdReplayMethod.__init__`） | 8 个 seed 456 快照都是 `monitor: train_holdout`。`metrics.jsonl` 每一段都是 `train.monitor=train_holdout`、`train.eval_split_read_for_selection=false`、`train.on_eval_end_ignored=true` | 留出路径已不用官方遗忘选样。旧 `v3_*` 的 `eval_split_monitor` 保持原协议，不改标、不重跑 |
| 2. 分数来自训练集、留出、验证还是最终测试 | `methods/pgdr.py` `split_train_holdout`、`score_holdout_map`；`core/evaluate.py` `_eval_segment` | 流里没有单独的 validation。监控集是每段 train 留出的 5 条。官方 eval 仍是每段 10 条，只给报告用。`score_holdout_map` 把这 5 条放进临时 `Segment` 的 `eval` 字段，`train` 为空，段名 `train_holdout_{id}`，再调用 `_eval_segment`。`_eval_segment` 只读取传入对象的 `segment.eval` | 字段名叫 `eval`，容易被误读成官方 test 文件 | `score_holdout_map` 构造 `train=[]`、`eval=examples`。两个 multiref 的 train/eval `(instruction, input)` 重叠都是 0。冒烟审计要求 holdout 键是 train 的子集且与官方 eval 不相交 | 这是复用打分函数，不是读取官方 test。论文和表里写“训练留出”，不要写成 test |
| 3. 选择器能否读到最终测试的标签或正确率 | `methods/pgdr.py` `select_forgotten`；`CeOpsdReplayMethod._select_forgotten`、`on_eval_end` | `select_forgotten` 只接收调用者传入的 `forgetting_by_segment`，自己不打开数据文件。`train_holdout` 时该字典由 `_refresh_holdout_forgetting` 写成留出正确率。`on_eval_end` 在该模式下于读取 `forgetting_by_segment` 之前返回 | `eval_split_monitor` 分支仍会把官方遗忘抄进选择器。`load_memory_dump` 也会装入 dump 里的遗忘字典；这 8 个快照没有 `replay.init_memory_path` | 代码分支加上 8 个 run 的 `on_eval_end_ignored=true`。单测 `test_holdout_never_enters_later_replay_and_eval_is_unread` 与 `test_eval_split_monitor_still_reads_on_eval_end` | 留出路径的选样看不到官方正确率。不要把新 run 配成 `eval_split_monitor`，也不要加载旧 dump 当记忆 |
| 4. 能否看到未来任务 | `CeOpsdReplayMethod.train_on_segment`、`_mix_current_and_replay`；`select_forgotten` | 当前段的记忆要等本段训练结束才 append 到 `_memories`。混合回放只读已经 append 的历史记忆。`select_forgotten` 只在这些记忆的 `segment_id` 里取 argmax | 未发现未来段进入回放池 | 段 0 的 `train.n_replay=0`（8 个全流程和两段冒烟都是）。之后才有回放 | 不改。未来任务没有进入选样 |
| 5. 监控是否只用回放决策之前已有的信息 | `CeOpsdReplayMethod.train_on_segment` 与 `core/train.py` 里 `evaluate_stream` 之后的 `on_eval_end` | 段 t 一开始就用段 t-1 结束时写下的留出遗忘来选回放。留出打分发生在本段优化器步骤之后、官方 `evaluate_stream` 之前。官方 eval 即使随后运行，`on_eval_end` 也不覆盖这份字典。本段自己的留出分要等本段训完才写入，不参与本段回放 | 留出标签会决定以后从哪一段记忆里抽样。这些标签来自未参与梯度的训练留出，不是官方 test | 调用顺序：`_training_examples` → 父类 `train_on_segment` 的梯度 → append 记忆 → `_refresh_holdout_forgetting` → 返回后 `core/train.py` 才 `evaluate_stream` 并调用 `on_eval_end` | 选样时序成立，不改。报告用的官方遗忘和选样用的留出遗忘是两套数 |
| 6. 监控划分是否在持续训练开始前就确定 | `methods/pgdr.py` `split_train_holdout` | 划分在每段 `_training_examples` 里现算，没有事先落盘。局部 RNG 只依赖 `(holdout_seed, segment_id)`：`Random((seed * 1000003) ^ (segment_id * 9176) ^ 0x5A17)`，不消耗训练 RNG，不读模型或分数 | 不是预处理文件，但是流顺序固定时，梯度开始前就可复现。它不随训练结果改变 | 函数注释与实现。新单测对段 0/7/18 检查了这一点 | 保持该函数。16 个新 run 继续用 `holdout_seed: 456`，划分与已完成的 seed 456 相同 |
| 7. 监控样本是否排除在梯度训练之外 | `CeOpsdReplayMethod._training_examples`；父类 `baselines/basic_baselines/ce_opsd/method.py` `train_on_segment` | 返回值是 fit 加历史回放，不是 `segment.train` 全文。CE 和 OPSD 都只遍历这个列表。`epochs_per_segment: 1`，每段只过一遍。审计字段 `holdout_in_train` 用对象 id 检查留出是否出现在混合列表 | `eval_split_monitor` 会训练全部 `segment.train` | 8 个 run 全部 `holdout_in_train=false`。`segment_001/train_metrics.json`（`v3h_proto_opsd_instrdialog_seed456`）里 `fit_n=45`、`n_train_examples=53`（45+8） | 留出不进损失。训练条数因此从 50 变为 45，必须单独报告，不能把样本数差异说成纯泄漏效应 |
| 8. 监控样本是否排除在历史记忆之外 | `CeOpsdReplayMethod._build_memory`、`_pick_diverse_k` | `train_holdout` 把 `source_examples=fit` 传入。`buffer_k=10` 只从这 45 条里轮转抽取。留出对象不在记忆列表里 | 若不传 `source_examples`，`_build_memory` 会退回整段 `segment.train`。留出路径没有走这个默认 | 8 个 run 全部 `holdout_in_memory=false`。快照 `buffer_k: 10`。两个 benchmark 每段 train 的 `(instruction, input)` 都是 50 个唯一键，去掉 5 条后仍够抽出 10 条 | 记忆 K=10 建在切分之后。磁盘没有逐条记忆清单（`save_segment_adapters: false`），10 是代码路径容量，不是从 dump 清点出来的条数。不改 K |
| 9. 监控样本会不会从另一条数据加载再进训练 | `CeOpsdReplayMethod._teacher_prototypes_for`；`core/models/base_model.py` `generate` / `sample_rollouts`；`score_holdout_map` | 因子格 `teacher_proto.mode: none` 时，教师原型函数在抽样前返回 `None`。官方 eval 和留出打分都是 `torch.no_grad` 生成，不把样本追加进训练列表。没有第二份训练 DataLoader | 若以后把 `teacher_proto` 改成 `random` 或 `matched`，记忆文本会进教师提示。记忆本身仍来自 fit，留出对象仍不在其中；那不是这 8 个 run 的配置 | 8 个快照 `teacher_proto.mode=none`，且无 `init_memory_path`。冒烟 `holdout_in_train=false` | 因子格路径没有第二条入口。不改 |
| 10. 划分是否固定、可复现 | `split_train_holdout`；`CeOpsdReplayMethod.__init__` 的 `holdout_seed`；`scripts/write_v3_configs.py` | 同一 `(holdout_seed, segment_id)` 得到同一下标。方法用的是 `replay.holdout_seed`，不是训练 `seed`。若 yaml 缺 `holdout_seed`，初始化会退回 `cfg["seed"]` | 退回训练 seed，或用 `write_v3_configs.py` 原样生成，都会让 789/42 的划分偏离 seed 456，四个方法也不再共用同一留出 | 新单测：训练 seed 为 789 和 42、`holdout_seed` 仍为 456 时，`_training_examples` 的 `holdout_indices` 与 `split_train_holdout(..., seed=456)` 相同。既有测试已证明把 789 当作 holdout seed 时划分不同 | 不把 `holdout_seed` 耦合到训练 seed。16 个新 yaml 把 `holdout_seed` 钉在 456。本次不改退回逻辑，因为 8 个已完成 yaml 都显式写了 456 |
| 11. 配对比较里的四个方法是否用同一批监控样本 | `split_train_holdout`；`tests/test_holdout_replay_opsd.py` `test_split_is_shared_across_the_four_methods` | 划分不读 `strategy` 或 `lambda_opsd`。uniform/proto、λ=0/0.2 在同一 `holdout_seed` 下下标相同 | 只有有人把每个方法或每个训练 seed 写成不同的 `holdout_seed` 时才会拆开 | 既有单测四个方法的下标集合大小为 1。新单测覆盖训练 seed 789/42 | 配对比较继续共用 `holdout_seed=456`。这是可比条件，不是可选项 |
| 12. 最终测试是否只用于报告，而不控制训练或挑选 checkpoint | `core/train.py`：`evaluate_stream`、`save_final_adapter`、段循环末尾的硬停止；`core/evaluate.py` `evaluate_stream` | 每段结束后仍对官方 `segment.eval` 做 11 指标，写入 `final.eval.*`。最终 adapter 是最后一段的权重，没有按官方分数挑 checkpoint。`trajectory_early_stop` 与 `ours_v10_early_stop` 在快照里缺键，函数默认关闭 | 硬停止不是选样，但读了官方分数。`train.hard_early_stop_enabled` 在 8 个快照里都不存在，`core/train.py` 默认 `True`。段 id ≥ 3 且 `seen_avg_score < 0.2` 时 `sys.exit(1)` | 8 个全流程分别有 19 或 38 段，说明该阈值没有触发。已入表的留出 Acc 最低是 InstrDialog 随机 CE 的 0.321 | 不关这个默认，否则与 8 个已验证 run 不是同一停止规则。它不写入 `_forgetting_by_segment`。新 run 若掉到 0.2 以下会停，应照实记录，不要为了跑完去改阈值 |
| 13. 历史测试结果是否用于回放优先级、超参或停止 | 回放：`on_eval_end` 与 `_select_forgotten`。超参：8 个 yaml。停止：上一条 | `train_holdout` 的回放优先级来自留出遗忘，不来自官方测试。λ、学习率、K、比例都写死在 yaml 里，训练循环不用本 run 的测试分去改它们。停止决定见第 12 条。`eval_split_monitor` 仍会用官方遗忘决定退化段 | Protocol A 的选样泄漏还在代码里，只是这批 `v3h_*` 没走它。硬停止是残留的测试反馈，作用是中止进程，不是给回放排序 | 8 个 run 的 λ 为 0 或 0.2，`replay_ratio=0.2`，`buffer_k=10`，学习率 `2e-4`，且选样标志全程为 false / ignored | 选样泄漏在 `train_holdout` 已消除。不把 Protocol A 行放进这张 24 行表。不因硬停止去改历史协议。GPU 阶段可以开始 |

## `score_holdout_map` 不是官方 test

`score_holdout_map`（`methods/pgdr.py`）对每个已见段做三件事：

1. 取出该方法自己保存的训练留出样本。
2. 造一个临时 `Segment`：`segment_name` 为 `train_holdout_{sid}`，`train` 为空列表，`eval` 等于这些留出样本。
3. 调用 `core.evaluate._eval_segment`。该函数用 `segment.eval` 组 prompt、生成、再按多 gold 取最佳对，返回 strict accuracy。

字段名是 `eval`，因为 `_eval_segment` 的打分入口就叫这个。内容是 `split_train_holdout` 从 `segment.train` 划出去的行。函数不接收官方 eval 列表，也不打开 `benchmark/*_multiref.json` 的 `eval`。`_refresh_holdout_forgetting` 的注释写明它不接收 `segment.eval`。

官方 test 的另一条链在 `core/train.py`：训练返回之后，`evaluate_stream` 对真实 `segments_seen` 的 `segment.eval` 打分。随后 `on_eval_end` 在 `train_holdout` 下返回，不读 `extra.forgetting_by_segment`。因此官方正确率可以出现在日志和 `final.eval` 里，但不会成为下一段的退化槽分数。

留出打分用的是 strict match 比例（`_eval_segment` 的第一个返回值）。任务感知准确率被丢掉。选择器排序与论文报告的 Forgetting 不是同一个数。

## 数据核算

计数来自：

- `benchmark/instrdialog/citb_cl_dialogue_tasks_train50_eval10_multiref.json`
- `benchmark/instrdialogpp/citb_cl_38_random_tasks_train50_eval10_multiref.json`

逐段统计：InstrDialog 19 段、InstrDialog++ 38 段，每段 train 都是 50 行，eval 都是 10 行，train 的 `(instruction, input)` 都唯一，train 与 eval 的键重叠都是 0。多 gold 在同一行的 `outputs` 里：InstrDialog train 有 40 行多于一条 gold，InstrDialog++ 有 155 行。下表只计行，不把多 gold 拆成多行。

`Replay Memory N` 是切分之后从 fit 建立的 `buffer_k`，不是本段混进损失的回放条数。每段唯一键足够抽出 10 条。run 目录没有逐条记忆 dump，不能再做一次落盘清点。

`Trainable Token Count` 不能从已有 metrics 精确还原，表内不填估算值。缺口见下节。

| Benchmark | Segment | Original Train N | Held-out Monitor N | Remaining Train N | Replay Memory N | Trainable Token Count |
|---|---|---|---|---|---|---|
| InstrDialog | 0–18，逐段相同 | 50 | 5 | 45 | 10 | 不能从已有 metrics 精确还原 |
| InstrDialog | 19 段合计 | 950 | 95 | 855 | 190 | 不能从已有 metrics 精确还原 |
| InstrDialog++ | 0–37，逐段相同 | 50 | 5 | 45 | 10 | 不能从已有 metrics 精确还原 |
| InstrDialog++ | 38 段合计 | 1900 | 190 | 1710 | 380 | 不能从已有 metrics 精确还原 |

官方 eval 不在这张表里：InstrDialog 19×10=190，InstrDialog++ 38×10=380。它不进入 fit，也不进入记忆。

### Token 计数的来源和缺口

能精确加总的是优化器 step，来自各 run 的 `metrics.jsonl` 字段 `train.train.n_steps`（与 `train.batches` 相等）。batch size 是 2，每段 1 个 epoch。

| Run | 段数 | 段 0 step | 其后每段 step | step 合计 | 段 0 回放条数 | 其后每段回放 | 回放合计 |
|---|---|---|---|---|---|---|---|
| `v3h_rand_ce_instrdialog_seed456` | 19 | 23 | 27 | 509 | 0 | 9 | 162 |
| `v3h_proto_ce_instrdialog_seed456` | 19 | 23 | 27 | 509 | 0 | 8 | 144 |
| `v3h_rand_opsd_instrdialog_seed456` | 19 | 23 | 27 | 509 | 0 | 9 | 162 |
| `v3h_proto_opsd_instrdialog_seed456` | 19 | 23 | 27 | 509 | 0 | 8 | 144 |
| `v3h_rand_ce_instrdialogpp_seed456` | 38 | 23 | 27 | 1022 | 0 | 9 | 333 |
| `v3h_proto_ce_instrdialogpp_seed456` | 38 | 23 | 27 | 1022 | 0 | 8 | 296 |
| `v3h_rand_opsd_instrdialogpp_seed456` | 38 | 23 | 27 | 1022 | 0 | 9 | 333 |
| `v3h_proto_opsd_instrdialogpp_seed456` | 38 | 23 | 27 | 1022 | 0 | 8 | 296 |

段 0 的 23 步等于 `ceil(45/2)`。其后 uniform 的混合长度是 45+9=54，proto 是 45+8=53，二者 `ceil(/2)` 都是 27。step 数相同，但 proto 的最后一个 batch 只有 1 条，uniform 是 2 条。

不能还原 token 的原因：

- CE 监督 token 长度没有写入 `train_metrics.json` 或 `metrics.jsonl`。每步从该行 gold 里 `random.choice` 一条，当时抽中的 gold 没有记录，现在重分词对不上那次抽样。
- OPSD 只留下 `train.rollout_mean_tokens`。`run_opsd_step` 的 `rollout_token_total` 和 `rollout_count` 没有进入保存的 metrics。空续写会被跳过，所以不能用“均值 × 混合条数”回推总数。CE-only 的该均值是 0，因为 `lambda_opsd <= 0` 时不采样（日志 `opsd_skipped_lambda=true`、`opsd_applied=false`）。
- 留出监控另有约 5 条/段的生成，只用于打分，不是可训练 token。

### 配额 9 与 8 是已验证协议

`methods/pgdr.py` `quota` 使用 `int(round(n * ratio))`。`n_current=45` 时：

- uniform：`round(45 * 0.2) = round(9) = 9`，近邻槽和退化槽置 0。
- proto：`round(45 * 0.1) + round(45 * 0.1)`。Python 3 对 `.5` 向偶数舍入，`round(4.5) = 4`，所以是 4+4=8。

8 个全流程日志与此一致，没有第三种回放条数。两槽在 `_mix_current_and_replay` 里直接拼接，不去重。proto 上 `unique_pairs < n_replay` 的段：InstrDialog proto CE 为段 1、2、3、9；proto OPSD 为段 1、2、3、9、15；InstrDialog++ proto CE 为段 1、2、5、6、7、9、26；proto OPSD 为段 1、2、5、7、9、11。四个 uniform run 的该差为 0。有效唯一回放可以少于 8。本次不改 `replay_quota`，否则 seed 456 的 8 个 run 不再可比。论文不能写随机与 PGDR 的回放条数逐条相同。

### 不要与 Protocol A 混表

Protocol A 不留出，每段训练 50 条。同一配额函数下 uniform 为 `round(50 * 0.2) = 10`，proto 为 `round(5) + round(5) = 10`。那是另一套训练条数和另一套回放条数。旧 `v3_*` 还用官方 eval 遗忘选样。上表只描述 Protocol B。A 与 B 的 Acc/F 差同时包含泄漏差异和 50→45 的样本差异。

### 近邻槽不是余弦检索

`CeOpsdReplayMethod._select_neighbors` 先按任务类型和 gold 长度做格式兼容，用冻结指令向量的点积把历史段排序，然后把这些段的记忆全部放进一个池，再 `random.sample`。排序不改变无放回均匀抽样。默认 `transfer_slot: neighbor` 的 8 个 proto run 都是这条路径。`transfer_slot: anti` 才会取余弦最低的一段，因子格没有用它。

这不是测试泄漏。改成余弦检索会变成新协议，8 个 seed 456 作废。论文不能把近邻槽写成余弦最近邻。

## 教师、损失和 11 个指标

OPSD 教师前向在 `baselines/basic_baselines/opsd/method.py` `_teacher_forward_logits`。`teacher_source="base"` 时包在 `torch.no_grad()` 里，并对 PEFT 模型调用 `disable_adapter()`。学生 logits 在这个上下文之外由 `CeOpsdMethod.train_on_segment` 计算。`run_opsd_step(..., do_step=False)` 只返回损失张量，真正的 `backward` 和 `step_adapter` 只有一次，对象是 `ce_plus_opsd` 的 CE + λ·JSD。

JSD 在 `methods/opsd.py` `token_divergence`：`m = β p_T + (1-β) p_S`，`β KL(p_T‖m) + (1-β) KL(p_S‖m)`，β=0.5，对续写 token 再对样本取均值。没有单独的蒸馏温度。学生续写来自 `sample_rollouts`（`do_sample=True`）。8 个快照没有写 `student_tokens`，`CeOpsdMethod` 默认 `on_policy`。教师参考句是训练 gold 里与续写 Token F1 最高的一条，不是 eval gold。λ=0 时不构造 OPSD 图。

8 个快照的其余训练设置相同：LoRA r=16、alpha=32、`q_proj`/`v_proj`、lr=2e-4、batch 2、每段 1 epoch、`format_normalize: true`、数据为上述 multiref、最大段数为全部。`model.max_seq_len` 未写，模型默认 2048。序列长至少 1024 的样本跳过 OPSD；这 8 个 run 的 `opsd_skipped_long` 为 0。没有梯度累积。优化器是 `LoRAWrapper._rebuild_optimizer` 里的 `AdamW`，学习率在 `step_adapter` 里改成当前 lr。yaml 的 `weight_decay: 0.0` 和 `grad_clip_norm: 1.0` 没有被训练代码读取；AdamW 因此使用 PyTorch 默认 `weight_decay=0.01`，也没有梯度裁剪。这是 8 个已完成 run 的实现，本次不改，避免新 seed 和它们分叉。

遗忘公式在 `core/evaluate.py` `evaluate_stream`：非当前段 `max(0, 历史最佳 - 当前)` 的平均，当前段记 0 且不进平均。任务感知遗忘用同一形状。多 gold 在 `_best_pred_gold_metrics`：任一 gold EM=1 优先，否则取 token F1 最大的一对，task-aware、ROUGE-L、BLEU、LCS 跟这一对。`format_normalize: true` 时 pred 与 gold 对称归一。11 个飞书列对应 `final.eval` 的 `seen_avg_score`、`seen_avg_task_aware_score`、`forgetting`、`task_aware_forgetting`、`token_f1_mean`、`rouge_l_mean`、`bleu_mean`、`lcs_overlap_mean`、`current_score`、`current_task_aware_score`、`task_aware_score_mean`。

## 测试与已有冒烟

`PYTHONPATH=/root/research-repos/opd-lora_v3`，解释器 `/root/autodl-tmp/conda_envs/rp_lora_v0/bin/python`，在仓库根运行 `tests/test_holdout_replay_opsd.py`：18 项通过，含新增的 `test_training_seed_789_and_42_keep_holdout_seed_456_split`。该测试确认训练 seed 为 789 和 42 时，只要 `holdout_seed` 仍是 456，划分与 seed 456 相同。

`scripts/check_v3h_smoke.py` 复查 `/root/autodl-tmp/rp_lora_v3_runs/runs/v3h_proto_opsd_instrdialog_seed456_smoke2seg`：输出 `SMOKE_HOLDOUT_OK`，退出码 0。该目录只有两段，不与全流程比 Acc。脚本按既有行为刷新了 `/root/autodl-tmp/rp_lora_v3_runs/smoke_audit.json`（`ok: true`）。没有改 run 目录里的训练产物。

## 放行

泄漏进入选样这一条，已在 `train_holdout` 消除。训练样本数相对 Protocol A 有改变。GPU 阶段可以开始 16 个新 run。

开始前配置必须同时满足：`monitor: train_holdout`，`holdout_seed: 456`，训练 seed 仅为 789 或 42，数据与 `format_normalize` 与上述 8 个 run 相同。不要用未改过的 `write_v3_configs.py` 生成它们。

不要为了让随机和 PGDR 的回放条数相同去改 `quota`。不要把近邻槽改成余弦检索。不要把留出样本加回训练集。不要把官方 eval 遗忘重新接进 `on_eval_end`。
