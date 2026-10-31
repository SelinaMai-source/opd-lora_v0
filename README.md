# opd-lora v3

Staging repo for the PGDR continual-learning runs. Training entry point:

```bash
python -m core.trainer --config configs/<run>.yaml
```

Weights and benchmark jsons are symlinks to the data disk (`assets/pretrained`, `benchmark/`). Run outputs go to `/root/autodl-tmp/rp_lora_v3_runs`. Checkpoints are not stored in git.

`replay.monitor`:

- `eval_split_monitor`: old protocol. Forgotten replay uses official-eval forgetting from `on_eval_end`.
- `train_holdout`: each segment holds out 5 of 50 train rows with `holdout_seed`. Those rows are excluded from CE and from the replay buffer. Forgetting for the forgotten slot is computed on that holdout. The four `v3h_` methods share the split. Official eval is still the reported metric and is not read for selection.
