"""Figures for realistic-1000 failure analysis (English labels to avoid CJK font issues)."""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_realistic1000_gpt_image2')
rep = json.load(open(OUT / 'analysis_report.json', encoding='utf-8'))
rows = list(csv.DictReader(open(OUT / 'perstream_full496.csv', encoding='utf-8-sig')))

streams = ['clip', 'fft', 'dct', 'dire', 'noise']

# ---- Fig 1: per-stream hit vs miss mean prob + gap ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
hit = [rep['hit_vs_miss_stream_contrast'][s]['hit_mean'] for s in streams]
miss = [rep['hit_vs_miss_stream_contrast'][s]['miss_mean'] for s in streams]
x = np.arange(len(streams))
ax[0].bar(x - 0.2, hit, 0.4, label='Caught (fusion=AI)', color='#2a9d8f')
ax[0].bar(x + 0.2, miss, 0.4, label='Missed (fusion=Real)', color='#e76f51')
ax[0].axhline(0.5, ls='--', c='gray', lw=1)
ax[0].set_xticks(x); ax[0].set_xticklabels([s.upper() for s in streams])
ax[0].set_ylabel('mean prob(AI)'); ax[0].set_ylim(0, 0.8)
ax[0].set_title('Per-stream mean confidence: caught vs missed')
ax[0].legend()
for i, (h, m) in enumerate(zip(hit, miss)):
    ax[0].text(i, max(h, m) + 0.02, f'gap\n{h-m:+.2f}', ha='center', fontsize=9)

rate = [rep['per_stream'][s]['detect_rate'] for s in streams] + [rep['per_stream']['fusion']['detect_rate']]
labels = [s.upper() for s in streams] + ['FUSION']
colors = ['#264653'] * 5 + ['#e9c46a']
ax[1].barh(labels, rate, color=colors)
ax[1].set_xlabel('detection rate (%)'); ax[1].set_title('Detection rate per stream (alone)')
for i, v in enumerate(rate):
    ax[1].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUT / 'fig_perstream_gap.png', dpi=150)
print('[saved] fig_perstream_gap.png')

# ---- Fig 2: fusion_prob distribution + CLIP scatter ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fp = np.array([float(r['fusion_prob']) for r in rows])
ax[0].hist(fp, bins=40, color='#457b9d', edgecolor='white')
ax[0].axvline(0.5, ls='--', c='red', lw=1.5, label='AI threshold 0.5')
ax[0].set_xlabel('fusion prob(AI)'); ax[0].set_ylabel('# images')
ax[0].set_title(f'Fusion score distribution (only {rep["overall_detect_rate"]}% cross 0.5)')
ax[0].legend()

clip = np.array([float(r['clip_prob']) for r in rows])
ax[1].scatter(clip, fp, s=12, alpha=0.5, color='#6a4c93')
ax[1].axhline(0.5, ls='--', c='red', lw=1); ax[1].axvline(0.5, ls='--', c='red', lw=1)
ax[1].set_xlabel('CLIP stream prob(AI)'); ax[1].set_ylabel('fusion prob(AI)')
ax[1].set_title('Fusion tracks CLIP almost 1:1')
plt.tight_layout()
plt.savefig(OUT / 'fig_score_dist.png', dpi=150)
print('[saved] fig_score_dist.png')

# correlation fusion vs each stream
print('\nPearson corr  fusion_prob vs stream:')
for s in streams:
    sv = np.array([float(r[f'{s}_prob']) for r in rows])
    c = np.corrcoef(fp, sv)[0, 1]
    print(f'  {s:6s}  r = {c:+.3f}')
