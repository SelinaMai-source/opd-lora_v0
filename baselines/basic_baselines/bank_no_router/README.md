# bank_no_router

`bank_no_router` is one of the project's original basic baselines, but it is implemented as a configuration branch in the unified training pipeline rather than as a standalone method directory.

Expected use:

```bash
python core/train.py --config configs/baseline.yaml
```

Set the baseline/method option in the config to the bank-without-router variant. This baseline keeps the LoRA bank capacity while removing the learned routing component, so it is mainly used to isolate the contribution of the router.

Status: collected here as a documentation placeholder. No training code was duplicated or moved for this variant.
