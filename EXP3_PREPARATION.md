# EXP3 Hybrid Preparation

## 目的

本文件是 Hybrid（DCT + CLIP + DINOv2、Self-Attention + Cross-Attention）進入 EXP3 前的準備清單。EXP3 的模型與 threshold 必須先凍結；測試結果不能反過來用於挑模型或調參。

## 目前可確認的前提

- Final candidate: DCT + CLIP + DINOv2。
- Fusion candidate: Hybrid。
- Existing checkpoint: `outputs/exp2a/hybrid/best_model.pth`。
- Existing evaluation cache: `handoff_dct_clip_dinov2_20260826/feature_data/crop`。
- Current cache has `train`, `val`, and `cross_generator_test`; it does not contain a separate seen-generator holdout, multi-severity corrupted-image cache, or open-world external set.
- `locked_test` remains unopened.

## EXP3 execution order

1. EXP3A: basic detection on a fixed held-out split.
2. EXP3B: seen/unseen generator evaluation, with each generator reported separately.
3. EXP3C: run only if reliable content labels exist.
4. EXP3D: regenerate multi-severity JPEG, resize, blur, and noise features from the original images.
5. EXP3E: calibration and confidence analysis without retraining.
6. EXP3F: compare the frozen Hybrid model with the selected baselines below.
7. EXP3G: evaluate a truly new external/open-world dataset after all settings are frozen.
8. Use `locked_test` once, at the end.

## SOTA comparison set

### Primary set

| Method | Venue/year | Why it belongs | Reproduction route | Priority |
|---|---|---|---|---|
| UnivFD | CVPR 2023 | Widely used universal detector and a clean frozen-CLIP + classifier reference. | Official code/weights if compatible; otherwise document the exact released checkpoint. | Required |
| DIRE | ICCV 2023 | Representative diffusion-reconstruction detector with a distinct signal from our DCT/CLIP/DINOv2 fusion. | Official code and released weights; report its required diffusion reconstruction cost. | Required |
| NPR | CVPR 2024 | Pixel-neighborhood artifact detector; directly represents a low-level artifact family complementary to semantic features. | Official implementation or released model. | Required |
| FatFormer | CVPR 2024 | Generalizable CLIP-based adaptive transformer that explicitly combines visual and frequency forgery cues. | Official repository and released checkpoint. | Required |
| AIDE | ICLR 2025 | Recent hybrid-feature detector evaluated on GenImage and Chameleon; combines patchwise and semantic experts. | Official repository/checkpoint if compatible with the dataset protocol. | Required |

### Reference and optional set

| Method | Role | Reason not in the minimum set |
|---|---|---|
| CNNSpot / CNNDetection | Historical anchor | Important for continuity, but CVPR 2020 and not a recent SOTA claim. Include as a classical baseline, not as the only comparator. |
| FakeInversion | Specialized modern comparator | CVPR 2024 and strong unseen text-to-image focus, but requires Stable Diffusion inversion and substantially different computation. Include only if its data and environment can be reproduced fairly. |

### Selection rule

A method enters the final table only if its paper has a peer-reviewed venue or a clearly documented benchmark, it is cited/reused by later comparison studies, and its official code or weights can be identified. A paper's reported accuracy is not copied into our table; every included method must be evaluated under our declared protocol or marked as paper-reported and kept in a separate column.

## Fair comparison protocol

- Use the same real/fake test images whenever a method accepts the same input format.
- Keep the training data, generator visibility, and threshold policy explicit for every method.
- Separate `official pretrained` from `retrained under our protocol`; never mix them in one ranking.
- Report AUROC, AP, FPR, FNR, balanced accuracy, and ECE. Accuracy alone is insufficient.
- Report parameters, FLOPs, peak memory, and per-image latency with hardware and batch size.
- Do not give our model more training images than a baseline without recording it.
- Do not tune any baseline on `cross_generator_test` or `locked_test`.
- If a method requires a different input size, preprocessing, or diffusion inversion, report that difference instead of hiding it.

## Data that must be prepared before running the full comparison

### EXP3B

Create a non-selection holdout containing both:

- seen fake generators, plus real images;
- unseen fake generators, plus the same real-image policy.

The current `cross_generator_test` is mainly real images plus evaluation-only fake generators, so it can report overall unseen performance but cannot provide a meaningful seen-fake AUROC comparison by itself.

### EXP3C

Add a reliable `content` field such as portrait, landscape, building, animal, and object. If the label source is not reliable or coverage is unbalanced, mark EXP3C not run rather than inferring content from filenames.

### EXP3D

Generate and cache multiple severity levels for JPEG compression, resize, Gaussian blur, and Gaussian noise. Keep the same image row identity across all three streams and record the transformation parameters in the index.

### EXP3F

Prepare a separate result row for each method with:

- method, paper, venue/year, code/weight source;
- training data and visible generators;
- input preprocessing;
- AUROC/AP/FPR/FNR/ECE;
- parameters/FLOPs/latency/peak memory;
- whether the value is reproduced or paper-reported.

## Current blockers

- No EXP3-specific implementation exists yet for the primary baselines.
- The old `s4f_sota_compare_v2.py` targets the superseded five-stream GRL model and must not be used as the final EXP3F runner.
- No separate seen-generator holdout is present in the current handoff cache.
- No content-label manifest is present.
- No multi-severity transformed feature cache is present.
- No post-freeze open-world dataset is present.

## Literature basis

- [GenImage (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f4d4a021f9051a6c18183b059117e8b5-Abstract-Datasets_and_Benchmarks.html) defines cross-generator and degraded-image evaluation tasks.
- [UnivFD (CVPR 2023)](https://arxiv.org/abs/2302.10174) is the universal frozen-feature reference.
- [DIRE (ICCV 2023)](https://openaccess.thecvf.com.cn/content/ICCV2023/html/Wang_DIRE_for_Diffusion-Generated_Image_Detection_ICCV_2023_paper.html) is the diffusion-reconstruction reference.
- [NPR (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.html) is the neighboring-pixel artifact reference.
- [FatFormer (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Forgery-aware_Adaptive_Transformer_for_Generalizable_Synthetic_Image_Detection_CVPR_2024_paper.html) is the adaptive CLIP/frequency-aware transformer reference.
- [AIDE (ICLR 2025)](https://arxiv.org/abs/2406.19435) is the recent hybrid-feature and challenging-realism reference.
- [FakeInversion (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Cazenavette_FakeInversion_Learning_to_Detect_Images_from_Unseen_Text-to-Image_Models_by_CVPR_2024_paper.html) is the optional inversion-based reference.

## Decision before execution

The recommended first EXP3F table is:

`Hybrid (ours) | UnivFD | DIRE | NPR | FatFormer | AIDE | CNNSpot anchor`

Add FakeInversion only after confirming that its inversion pipeline and compute budget can be reproduced. Do not use the old five-stream GRL comparison as evidence for the new Hybrid model.
