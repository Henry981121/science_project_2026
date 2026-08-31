"""Generate EXP3D transformed images from the immutable manifest.

This is preparation only. It does not extract features or modify source files.
Run it after reviewing the manifest and storage budget.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


def transform(image, condition, value, seed):
    if condition == 'jpeg':
        from io import BytesIO
        buf = BytesIO()
        image.save(buf, format='JPEG', quality=int(value))
        buf.seek(0)
        return Image.open(buf).convert('RGB')
    if condition == 'resize':
        w, h = image.size
        small = image.resize((max(1, round(w * value)), max(1, round(h * value))), Image.Resampling.LANCZOS)
        return small.resize((w, h), Image.Resampling.LANCZOS)
    if condition == 'blur':
        return image.filter(ImageFilter.GaussianBlur(float(value)))
    if condition == 'noise':
        arr = np.asarray(image).astype(np.float32)
        rng = np.random.default_rng(seed)
        arr = np.clip(arr + rng.normal(0.0, float(value), arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode='RGB')
    raise ValueError(condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--stop', type=int)
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    out = Path(args.out_dir)
    stop = len(df) if args.stop is None else min(args.stop, len(df))
    for i, row in df.iloc[args.start:stop].iterrows():
        src = Path(row['path'])
        target = out / str(row['condition']) / str(row['severity']) / f"{int(row['source_row']):06d}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        with Image.open(src) as image:
            image = image.convert('RGB')
            transformed = transform(image, row['condition'], row['severity_value'], int(row['source_row']))
            transformed.save(target, format='JPEG', quality=95)
        if (i - args.start + 1) % 1000 == 0:
            print(f'processed={i + 1}/{stop}')
    print(f'complete rows={stop - args.start} out={out}')


if __name__ == '__main__':
    main()

