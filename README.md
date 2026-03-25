# AI Image Detector — FusionDetector + GRL

**C.-H. Chao, H.-Y. Li, and C.-H. Shih**
*National Hualien Senior High School, Hualien, Taiwan*

---

A multi-stream AI-generated image detector using **Gradient Reversal Layer (GRL)** domain adversarial training. The model fuses 5 complementary feature streams (CLIP, FFT, DCT, DIRE, NoisePrint) through a Cross-Attention Transformer and learns generator-agnostic representations that generalise to unseen generators.

## Architecture

```
Image → 5 Feature Extractors (CLIP / FFT / DCT / DIRE / NoisePrint)
         ↓ (512-dim each)
    Cross-Attention Fusion (2-layer Transformer)
         ↓
    Shared Backbone (MLP, 512-dim)
    ├─ head_binary       → CE_binary   (Real / Fake)
    ├─ head_source  × λ  → CE_source   (generator identity, optional)
    └─ GRL → gen_discriminator → CE_gen_disc  (gradient-reversed)
```

**Loss:** `Total = CE_binary + λ_src × CE_source + λ_grl(t) × CE_gen`

**GRL schedule (Ganin et al., 2015):** `λ_grl(t) = λ_max × (2 / (1 + e^{-γt}) - 1)`

## Requirements

```bash
pip install torch torchvision scikit-learn pandas numpy pillow matplotlib scipy PyWavelets timm
# Optional: pip install grad-cam   (for Grad-CAM visualisation in s4_supplementary.py)
# Optional: pip install pywt       (for NSS experiment in s4h_nss_experiment.py)
```

## Setup

1. Edit `config.py` if your `data/` and `outputs/` folders are not inside the project root.
   By default everything is relative to the repo directory — **no changes needed** for most setups.

2. Prepare data splits (`data/splits/`):
   - `train.csv`, `val.csv`, `test.csv`, `cross_generator_test.csv`
   - Required columns: `path`, `label` (0=real, 1=fake), `generator`, `difficulty`

## Running Order

```bash
# Step 1 — Extract features & train single-stream baselines (EXP-A)
python s3a_single_stream.py

# Step 2 — Ablation: all stream combinations (EXP-B)
python s3b_ablation.py

# Step 3 — Fusion strategy comparison: Concat / Weighted / Cross-Attention (EXP-C)
python s3c_fusion_compare.py
python s3c_add_g2.py          # G2 supplement for EXP-C

# Step 4 — Main training: FusionDetector + GRL (produces best_model.pth)
python s3_main_grl.py

# Step 5 — Generalisation test G1/G2/G3 (EXP-D)
python s4d_grl_generalization.py

# Step 6 — SOTA comparison (EXP-F)
python s4f_sota_compare_v2.py

# Step 7 — Robustness test under image degradation (EXP-G)
python s4g_robustness_test.py

# Step 8 — NSS verification experiment (EXP-H)
python s4h_nss_experiment.py

# Step 9 — Multi-seed validation (EXP-I)
python s4i_multi_seed.py

# Step 10 — Supplementary visualisations (Grad-CAM, attention map, spectrum, error analysis)
python s4_supplementary.py all        # run all steps
python s4_supplementary.py 5 6 8 11  # run specific steps
```

## Key Hyperparameters (`s3_main_grl.py`)

| Parameter | Value | Description |
|---|---|---|
| `EPOCHS` | 30 | Training epochs |
| `BATCH_SIZE` | 256 | Batch size |
| `LR` | 1e-3 | Learning rate (AdamW) |
| `WARMUP_EPOCHS` | 5 | Linear LR warmup |
| `LAMBDA_GRL` | 0.05 | GRL loss weight (λ_max) |
| `LAMBDA_SRC` | 0.1 | Source loss weight |
| `GRL_GAMMA` | 10.0 | Progressive schedule growth rate |
| `CLIP_GRAD` | 1.0 | Gradient clipping |
| `d_model` | 512 | Attention dimension |
| `n_heads` | 8 | Attention heads |
| `n_layers` | 2 | Transformer layers |
| `dropout` | 0.3 | Backbone dropout |

## Output Structure

```
outputs/
├── exp_a/features/          # Cached stream features (.pt)
├── exp_a/{stream}/          # Per-stream classifier checkpoints
├── exp_b/results.json       # Ablation study results
├── exp_c/results.json       # Fusion comparison results
├── main_grl/
│   ├── best_model.pth       # Main GRL checkpoint
│   ├── training_history.json
│   └── final_results.json
├── exp_d_grl/               # Generalisation test results
├── exp_f_v2/                # SOTA comparison
├── exp_g/                   # Robustness test results
├── exp_h/                   # NSS experiment results
├── exp_i/                   # Multi-seed validation results
└── supplementary/           # Grad-CAM / attention / spectrum plots
```

## Generator Support

| ID | Generator | Type |
|---|---|---|
| 0–4 | ADM, GLIDE, SDv4, SDv5, Midjourney | Diffusion |
| 5–9 | WildFake, BigGAN, VQDM, Wukong, Firefly | Mixed |
| 10–11 | Real, Real_extra | Real |
| 12–18 | WildFake_DDIM, WildFake_other, StyleGAN, DCGAN, ... | GAN |

Generators with IDs 12–18 are treated as **unseen** in cross-generator (G2) evaluation.

## Demo

```bash
python ai_detector_demo.py --image path/to/image.jpg
```

## References

- Ganin et al. (2015) — [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)
- Wang et al. (2020) — [CNN-generated images are surprisingly easy to spot](https://arxiv.org/abs/1912.11035)
- Ojha et al. (2023) — [UnivFD: Towards Universal Fake Image Detection](https://arxiv.org/abs/2302.10174)
