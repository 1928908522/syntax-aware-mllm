# Result provenance

`experiment_results.json` records the aggregate values observed in the original research workspace. The plotting script reads this file directly and does not recompute model predictions.

Current records cover:

- Four negative-caption construction settings on approximately 10k MSCOCO samples.
- Partial DPO and 5% syntax-first RL on a 100-sample syntax-target held-out subset.

Raw prediction JSONL and checkpoints are intentionally excluded because they contain local dataset paths or large model states. The aggregate values are therefore suitable for documentation and visualization, but they are not a substitute for releasing raw predictions in a future reproducibility package.
