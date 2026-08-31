# EXP3D transformed-feature pipeline

## Required output topology

Each row in `cross_generator_multiseverity.csv` must produce one transformed image and three aligned feature rows:

`row_id -> transformed image -> DCT(192) + CLIP(768) + DINOv2(1024)`

Recommended output:

```text
exp3d_cache/
  images/<condition>/<severity>/<row_id>.jpg
  features/<condition>/<severity>/dct.npy
  features/<condition>/<severity>/clip.npy
  features/<condition>/<severity>/dinov2.npy
  index.csv
```

## Conditions

- JPEG: q95, q75, q50, q30
- Resize: scale0875, scale075, scale050, scale025
- Gaussian blur: sigma0_5, sigma1, sigma2, sigma3
- Gaussian noise: sigma2, sigma5, sigma10, sigma20

## Required checks before evaluation

- Every manifest row has exactly one transformed image.
- Feature row order matches `row_id`, `path`, `condition`, and `severity`.
- Shapes are DCT=192, CLIP=768, DINOv2=1024.
- No invalid or non-finite values.
- Original images remain unchanged.
- Transformations are applied only to evaluation images; no transformed image enters training.
- Report image count, missing count, invalid count, device, batch size, and extractor versions.

