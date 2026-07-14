"""
EXP-L 圖 X：成功 vs 失敗案例的 5 流預測對比
=============================================
凸顯「CLIP 一條流主導成敗」之核心發現
"""

import os, csv, json
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for fpath in (r'C:\Windows\Fonts\msjh.ttc', r'C:\Windows\Fonts\msyh.ttc'):
    if os.path.exists(fpath):
        fm.fontManager.addfont(fpath)
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

CSV_PATH = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L_chatgpt_image2\per_image_results.csv')
OUT      = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L_chatgpt_image2\figL_stream_comparison.png')
JSON_OUT = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L_chatgpt_image2\stream_comparison.json')

STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']
LABELS  = ['CLIP', 'FFT', 'DCT', 'DIRE', 'Noise']
COLORS  = {'success': '#3a8c5a', 'failed': '#a13030'}


def main():
    with open(CSV_PATH, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    succ = [r for r in rows if r.get('fusion_pred') == 'AI']
    fail = [r for r in rows if r.get('fusion_pred') == 'Real']

    print(f'成功 N = {len(succ)}, 失敗 N = {len(fail)}')

    # 計算每流 fake_prob 平均
    succ_fake  = {s: np.mean([float(r[f'{s}_fake'])  for r in succ]) for s in STREAMS}
    fail_fake  = {s: np.mean([float(r[f'{s}_fake'])  for r in fail]) for s in STREAMS}
    succ_w     = {s: np.mean([float(r[f'{s}_weight']) for r in succ]) for s in STREAMS}
    fail_w     = {s: np.mean([float(r[f'{s}_weight']) for r in fail]) for s in STREAMS}

    # 標準差（穩定性）
    succ_std   = {s: np.std([float(r[f'{s}_fake'])  for r in succ]) for s in STREAMS}
    fail_std   = {s: np.std([float(r[f'{s}_fake'])  for r in fail]) for s in STREAMS}

    # ─── 儲存 JSON 供報告書引用 ───
    payload = {
        'n_success': len(succ),
        'n_failed':  len(fail),
        'success': {'fake_mean': succ_fake, 'fake_std': succ_std, 'weight_mean': succ_w},
        'failed':  {'fake_mean': fail_fake, 'fake_std': fail_std, 'weight_mean': fail_w},
        'delta_fake': {s: round(succ_fake[s] - fail_fake[s], 2) for s in STREAMS},
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[saved] {JSON_OUT}')

    # ─── 繪圖：兩 panel ───
    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                              gridspec_kw={'width_ratios': [1.2, 1]})

    # ── Panel (a): 5 流 fake_prob 對比（成功 vs 失敗）──
    ax = axes[0]
    x = np.arange(len(STREAMS))
    w = 0.36

    succ_vals = [succ_fake[s] for s in STREAMS]
    fail_vals = [fail_fake[s] for s in STREAMS]
    succ_err  = [succ_std[s]  for s in STREAMS]
    fail_err  = [fail_std[s]  for s in STREAMS]

    bars_s = ax.bar(x - w/2, succ_vals, w, yerr=succ_err, capsize=4,
                    color=COLORS['success'], edgecolor='black', linewidth=0.7,
                    label=f'成功案例 (N={len(succ)})')
    bars_f = ax.bar(x + w/2, fail_vals, w, yerr=fail_err, capsize=4,
                    color=COLORS['failed'],  edgecolor='black', linewidth=0.7,
                    label=f'失敗案例 (N={len(fail)})')

    # 標數值
    for b, v in zip(bars_s, succ_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 3,
                f'{v:.1f}%', ha='center', fontsize=10.5, fontweight='bold',
                color='#1f5e30')
    for b, v in zip(bars_f, fail_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 3,
                f'{v:.1f}%', ha='center', fontsize=10.5, fontweight='bold',
                color='#7a2222')

    # 決策邊界
    ax.axhline(50, color='gray', ls='--', lw=1, alpha=0.6,
                label='決策邊界 = 50%')

    # CLIP 落差用紅色雙箭頭強調
    clip_drop = succ_fake['clip'] - fail_fake['clip']
    ax.annotate('', xy=(0 + w/2, fail_fake['clip']),
                xytext=(0 - w/2, succ_fake['clip']),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2.2))
    ax.text(0.55, (succ_fake['clip'] + fail_fake['clip']) / 2,
            f'落差\n{clip_drop:.1f}%',
            ha='left', va='center', fontsize=11, fontweight='bold', color='red',
            bbox=dict(facecolor='white', edgecolor='red',
                      boxstyle='round,pad=0.3', lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=12, fontweight='bold')
    ax.set_ylabel('Fake 機率 (%)', fontsize=12)
    ax.set_ylim(0, 125)
    ax.set_title(f'(a) 五流 Fake 機率對比\n成功案例 vs 失敗案例之 per-stream 預測',
                 fontsize=13, fontweight='bold', pad=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

    # ── Panel (b): Cross-Attention 權重對比 ──
    ax = axes[1]
    succ_wvals = [succ_w[s] for s in STREAMS]
    fail_wvals = [fail_w[s] for s in STREAMS]

    bars_s = ax.bar(x - w/2, succ_wvals, w,
                    color=COLORS['success'], edgecolor='black', linewidth=0.7,
                    label=f'成功案例 (N={len(succ)})')
    bars_f = ax.bar(x + w/2, fail_wvals, w,
                    color=COLORS['failed'],  edgecolor='black', linewidth=0.7,
                    label=f'失敗案例 (N={len(fail)})')

    for b, v in zip(bars_s, succ_wvals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.8,
                f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold',
                color='#1f5e30')
    for b, v in zip(bars_f, fail_wvals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.8,
                f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold',
                color='#7a2222')

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Attention 平均權重 (%)', fontsize=12)
    ax.set_ylim(0, max(max(succ_wvals), max(fail_wvals)) * 1.25)
    ax.set_title('(b) Cross-Attention 平均權重對比\n(CLIP 權重在成功 / 失敗皆 ~50%，無變化)',
                 fontsize=13, fontweight='bold', pad=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

    fig.suptitle(
        '圖（X） EXP-L：成功 vs 失敗案例之五流預測對比 — CLIP 為唯一具區辨力之關鍵流',
        fontsize=14, fontweight='bold', y=1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT}')

    # ─── 列印關鍵數字 ───
    print()
    print('=' * 70)
    print(f'  關鍵發現（{len(succ)} 成功 vs {len(fail)} 失敗）')
    print('=' * 70)
    print(f'  CLIP fake_prob:   成功 {succ_fake["clip"]:.1f}%  vs  失敗 {fail_fake["clip"]:.1f}%   '
          f'落差 {succ_fake["clip"] - fail_fake["clip"]:.1f}%')
    for s in ('fft', 'dct', 'dire', 'noise'):
        print(f'  {s.upper():>4s} fake_prob:   成功 {succ_fake[s]:.1f}%  vs  失敗 {fail_fake[s]:.1f}%   '
              f'落差 {succ_fake[s] - fail_fake[s]:+.1f}%')
    print()
    print(f'  CLIP weight (成功) = {succ_w["clip"]:.1f}%   |   CLIP weight (失敗) = {fail_w["clip"]:.1f}%')


if __name__ == '__main__':
    main()
