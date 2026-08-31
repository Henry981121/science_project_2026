# EXP3F same-data SOTA retraining

This is a new, isolated experiment batch. It does not overwrite the official
pretrained SOTA results or the current Hybrid result.

## Fixed data

- Train: 98,273 valid rows from `train_excluding_seen_holdout.csv`.
- Validation: the existing fixed validation manifest.
- Cross-generator test: the same valid 34,461-row manifest used by the
  previous official-checkpoint comparison.
- Seen-generator holdout: retained separately and never used for training.
- Locked test: not used in this phase.

## Fairness rule

All models receive the same image paths, labels, train/validation split, and
cross-generator test rows. Each model keeps only the preprocessing and
architecture-specific input transformation required by its paper. Best
checkpoints and thresholds are selected from validation only.

Models in scope: AIDE, DDA, UnivFD, and FerretNet. The project Hybrid result
is the fixed reference model.

## Required execution order

1. Verify manifests and image decodability.
2. Run one-batch sanity and short dry-run for each model.
3. Run full training with a fresh output directory and `training.log`.
4. Evaluate all models on the fixed cross-generator test only after training.
5. Run the locked test once, only after the model and threshold policy are
   frozen.

Official checkpoints remain available under `exp3f_sota_models_20260829` and
must not be used as the trained weights for this batch.
