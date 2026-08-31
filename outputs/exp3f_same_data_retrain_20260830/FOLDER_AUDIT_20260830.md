# Folder audit: 2026-08-30

## Decision

The same-data retraining batch is **not ready for full training yet**. The
data split audit is clean at the manifest/path level, but model-specific CSV
adapters and a complete image decode audit still must pass.

## Passed checks

- Train rows: 98,273.
- Validation rows: 23,438 validated rows.
- Independent cross-generator rows: 28,729 unique rows.
- Reserved seen-generator holdout rows: 10,920.
- Missing files: 0 in all four manifests.
- Exact path overlap between every pair of train, validation, cross-generator,
  and holdout manifests: 0.
- Project label convention verified: `fake=1`, `real=0`.
- Python compilation: 210 files passed.
- New batch contains no model checkpoint or result JSON, so no incomplete
  training result is being presented as final.

## Findings that are isolated, not silently repaired

- The former 34,461-row cross-generator source contained 5,732 duplicate
  paths. The new batch uses 28,729 unique rows.
- The former validation source had 23,474 rows, while the validated feature
  index used by the current Hybrid contains 23,438 rows. The new batch uses
  the validated 23,438-row index so the Hybrid reference remains aligned.
- Two legacy JSON groups are encoded in cp950 rather than UTF-8. They remain
  historical artifacts and are not read by the new batch.
- One legacy EXP2A log contains a Windows cp950 `UnicodeEncodeError` caused by
  printing the `⚠` marker. This is an old logging failure, not a model result.
- A long-running FerretNet `demo_image.py` process was found. It is unrelated
  to this batch and no training process is currently running.

## Retraining decision

The existing Hybrid checkpoint was already trained with the same 98,273-row
holdout-excluded training set. It does not need retraining solely to compare
against SOTA models. It does need a fresh evaluation on the new 28,729-row
cross-generator manifest. A retrain is required only if the training manifest,
label policy, or feature cache is changed.

The full image decode audit was started but stopped after exceeding the
interactive time budget. The source manifests already mark the training rows
as valid, and path/existence checks passed; do not treat this as a full decode
PASS. The next execution gate is a resumable, logged decode audit followed by
model-specific sanity and dry-run checks.
