# EXP3F same-data retraining batch

Status: data snapshot and protocol preparation complete; model training has
not been declared complete. AIDE, UnivFD, and FerretNet manifest data sanity
checks pass; DDA still requires paired VAE-reconstruction data.

## Fixed snapshot

The `manifests` directory contains the only manifests allowed for this batch:

- `train_excluding_seen_holdout.csv`: 98,273 rows, with fake label `1`.
- `val.csv`: 23,438 validated rows.
- `cross_generator_test_valid.csv`: 28,729 unique, independent rows.
- `seen_generator_holdout.csv`: 10,920 rows, reserved and never used for fit.
- `manifest_metadata.json`: counts, overlap checks, and source checksums.

The previous 34,461-row evaluation source contained 5,732 duplicate paths and
183 paths overlapping validation. Those rows were excluded here and are
recorded in `manifest_metadata.json` rather than silently discarded.

## Model protocol

Each model uses the same manifests and labels, while retaining its required
architecture-specific preprocessing:

| Model | Retraining policy | Input rule |
|---|---|---|
| Hybrid (reference) | Existing fixed retrained checkpoint | DCT + CLIP + DINOv2 |
| AIDE | Fine-tune released architecture from official initialization | Five-way DCT/SRM, 256px |
| DDA | Train DINOv2-LoRA detector | 336px, official DDA transforms |
| UnivFD | Train its linear detector head with CLIP backbone policy | 224px, official CLIP transform |
| FerretNet | Train detector using its local-pixel reconstruction pipeline | 256px, official FerretNet transform |

No model may read `cross_generator_test_valid.csv` during training or model
selection. Threshold calibration, if used, must use validation only.

## Hardware

The current machine exposes an NVIDIA GeForce RTX 5060 Laptop GPU with about
8 GB VRAM. Full AIDE and DDA fine-tuning may require reduced batch size,
gradient accumulation, or sequential execution. The dry-run must pass before
any full run is started.

## Current gate status

- Manifest split, label, and path checks: PASS.
- UnivFD data sanity: PASS, 98,273 rows, `(2, 3, 224, 224)`, labels `[0, 1]`.
- AIDE data sanity: PASS, 98,273 rows, `(5, 3, 256, 256)`, finite values.
- DDA: BLOCKED until scientifically valid paired real/VAE-reconstruction data
  and quality metadata are prepared.
- FerretNet data sanity: PASS, 98,273 rows, `(3, 256, 256)`, finite values.
- FerretNet startup: STOPPED after 200/3,071 batches of epoch 1 to avoid an
  impractical multi-day run; no checkpoint or result was produced.
- FerretNet full reconstruction/model forward: PENDING after speed optimization.
- `torch.compile` test: NOT AVAILABLE on this Windows environment because a
  working Triton installation is missing; the training config keeps compile
  disabled.
- Median LPD `kthvalue` benchmark: numerically identical but slower than the
  released median implementation, so it was not adopted.
- FerretNet batch-128 speedcheck: entered real forward/backward successfully,
  but reached about 7.8/8.1 GB VRAM and was stopped before checkpointing; this
  is a performance diagnostic only, not a model result.
- FerretNet full training remains blocked on practical runtime. No speedcheck
  checkpoint or metric is promoted into the experiment results.
- AIDE full retraining: INCOMPLETE. The fixed train/validation manifests and
  official ResNet-50 and ConvNeXt-XXLarge backbones were loaded with batch
  size 4, and `checkpoint-0.pth` was written after the first epoch. No later
  checkpoint or final metric is present, so this checkpoint is not promoted
  as a completed experiment result.
- Full training: AIDE INCOMPLETE; UnivFD PENDING; FerretNet BLOCKED on runtime.
