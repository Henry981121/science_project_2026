"""
Bar chart: per-generator D2 (Noise-ablation flip) rate.
Colored by family (Diffusion / Real / GAN).
"""

import os, sys
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\3.22output\level2\noise_decisive.json')
OUT = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\3.22output\level2\noise_decisive_bar.png')


# Family classification (mirrors preflight_risk_b.py logic)
FAMILY = {
    'real': 'Real', 'real_extra': 'Real',
    'dcgan': 'GAN', 'stylegan': 'GAN',
    'adm': 'Diffusion', 'glide': 'Diffusion', 'midjourney': 'Diffusion',
    'sdv4': 'Diffusion', 'sdv5': 'Diffusion', 'wildfake': 'Diffusion',
}
FAM_COLOR = {'Diffusion': '#C44E52', 'GAN': '#55A868', 'Real': '#4C72B0'}


def main():
    d = json.load(open(SRC, encoding='utf-8'))
    rows = d['per_generator']

    # Sort desc by D2 flip rate
    rows = sorted(rows, key=lambda r: -r['D2_flip_pct'])
    gens = [r['generator'] for r in rows]
    pcts = [r['D2_flip_pct'] for r in rows]
    cnts = [r['D2_flip_n']   for r in rows]
    ns   = [r['n']           for r in rows]
    cols = [FAM_COLOR.get(FAMILY.get(g, 'Other'), 'gray') for g in gens]

    # Overall D2 rate (weighted by sample size)
    overall_pct = d['definitions']['D2_flip']['pct']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(gens))
    bars = ax.bar(x, pcts, color=cols, edgecolor='black', linewidth=0.6)
    ax.axhline(overall_pct, color='black', linestyle='--', linewidth=1.2,
                label=f'overall = {overall_pct:.2f}%')

    # Value labels on top
    for b, p, c, n in zip(bars, pcts, cnts, ns):
        ax.text(b.get_x() + b.get_width() / 2,
                 p + 0.25,
                 f'{p:.2f}%\n({c}/{n})',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(gens, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('D2 flip rate (%)  — ablate Noise → prediction flips',
                  fontsize=11)
    ax.set_title('Per-generator rate: cases where removing Noise stream\n'
                  'flips the main model\'s prediction (crosses 0.5)',
                  fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(pcts) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # Family legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color=FAM_COLOR['Diffusion'], label='Diffusion'),
        Patch(color=FAM_COLOR['GAN'],       label='GAN'),
        Patch(color=FAM_COLOR['Real'],      label='Real'),
        plt.Line2D([0], [0], color='black', linestyle='--',
                   label=f'overall = {overall_pct:.2f}%'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=10)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[saved] {OUT}")


if __name__ == '__main__':
    main()
