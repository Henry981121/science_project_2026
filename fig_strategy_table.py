"""
圖（二十六）覆寫策略對照表 — 一策略一句話
"""

import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch

# CJK 字體
for fpath in (r'C:\Windows\Fonts\msjh.ttc',
              r'C:\Windows\Fonts\msyh.ttc',
              r'C:\Windows\Fonts\mingliu.ttc'):
    if os.path.exists(fpath):
        fm.fontManager.addfont(fpath)
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei',
                                            'PMingLiU', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

OUT = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig26_strategy_table.png')


# (marker_color, marker_shape, strategy_id, strategy_name, one_liner)
ROWS = [
    ('#222222', 'o', 'S0', 'Baseline',
     '只用主模型輸出，不啟用任何獨立信號（作為比較基準）。'),

    ('#C44E52', 's', 'S1', 'Energy Override',
     '當 Energy 分數超過閾值就強制判為 AI，否則交由主模型決定。'),

    ('#55A868', '^', 'S2', 'CA Override',
     '當色像差分數超過閾值就強制判為 AI，否則交由主模型決定。'),

    ('#8172B2', 'D', 'S3', 'OR （任一觸發即覆寫）',
     '只要 Energy 或 CA 任一信號超過閾值就判為 AI（最激進，FNR 最低但 FPR 最高）。'),

    ('#CCB974', 'v', 'S4', 'AND （雙信號同意才覆寫）',
     '必須 Energy 與 CA 兩信號同時超過閾值才判為 AI（最保守，誤判低但漏判高）。'),

    ('#4C72B0', 'P', 'S5', 'Tiered （階層式覆寫）',
     '先以高 Energy 閾值快速覆寫，未觸發者再用較寬閾值的 Energy + CA 聯合覆寫（高敏感場景首選）。'),

    ('#FF8C00', '*', 'S6', 'Vote （三信號軟投票）',
     '將三信號加權平均後再判決，不採覆寫機制（低誤判場景首選）。'),
]


def main():
    n_rows = len(ROWS)
    row_h  = 1.2
    col_w  = [0.7, 1.2, 5.8, 15.5]
    total_w = sum(col_w) + 0.6
    total_h = (n_rows + 1) * row_h + 2.0

    fig, ax = plt.subplots(figsize=(total_w * 0.62, total_h * 0.85))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis('off')

    # ── 主標題 ────────────────────────────────────────────────
    ax.text(total_w / 2, total_h - 0.55,
            '圖（二十六）覆寫策略對照表',
            ha='center', fontsize=20, fontweight='bold')
    ax.text(total_w / 2, total_h - 1.15,
            '對應圖（二十五）Pareto 之七種策略；marker 顏色形狀與 Pareto 一致',
            ha='center', fontsize=13, color='#555')

    # ── 表格起點 ──────────────────────────────────────────────
    x0 = 0.3
    y0 = total_h - 1.8

    # 計算各欄左邊起點
    col_x = [x0]
    for w in col_w[:-1]:
        col_x.append(col_x[-1] + w)

    # ── 表頭 ──────────────────────────────────────────────────
    hdr_y = y0 - row_h
    hdr_bg = Rectangle((x0, hdr_y), sum(col_w), row_h,
                        facecolor='#3a7d44', edgecolor='black',
                        linewidth=1.2, zorder=2)
    ax.add_patch(hdr_bg)
    headers = ['', '編號', '策略名稱', '說明']
    aligns  = ['center', 'center', 'left', 'left']
    for i, (txt, al) in enumerate(zip(headers, aligns)):
        tx = col_x[i] + (col_w[i] / 2 if al == 'center' else 0.25)
        ax.text(tx, hdr_y + row_h / 2, txt,
                va='center', ha=al,
                fontsize=15, fontweight='bold', color='white', zorder=3)

    # ── 資料列 ────────────────────────────────────────────────
    for i, (mk_c, mk_s, sid, sname, sentence) in enumerate(ROWS):
        row_y = hdr_y - (i + 1) * row_h

        # 斑馬紋背景
        bg_color = 'white' if i % 2 == 0 else '#f5f7fa'
        bg = Rectangle((x0, row_y), sum(col_w), row_h,
                        facecolor=bg_color, edgecolor='#ddd',
                        linewidth=0.6, zorder=1)
        ax.add_patch(bg)

        # 左邊色條
        sb = Rectangle((x0, row_y), 0.10, row_h,
                        facecolor=mk_c, edgecolor=None, zorder=2)
        ax.add_patch(sb)

        # 第 1 欄：marker
        ax.scatter([col_x[0] + col_w[0] / 2], [row_y + row_h / 2],
                    marker=mk_s, s=260, color=mk_c,
                    edgecolor='black', linewidth=1.1, zorder=3)

        # 第 2 欄：策略 ID
        ax.text(col_x[1] + col_w[1] / 2, row_y + row_h / 2, sid,
                va='center', ha='center',
                fontsize=16, fontweight='bold', color=mk_c, zorder=3)

        # 第 3 欄：策略名稱
        ax.text(col_x[2] + 0.2, row_y + row_h / 2, sname,
                va='center', ha='left',
                fontsize=14, fontweight='bold', color='#222', zorder=3)

        # 第 4 欄：說明
        ax.text(col_x[3] + 0.2, row_y + row_h / 2, sentence,
                va='center', ha='left',
                fontsize=13, color='#222', zorder=3)

    # 表格整體外框
    outer = Rectangle((x0, hdr_y - n_rows * row_h),
                       sum(col_w), (n_rows + 1) * row_h,
                       facecolor='none', edgecolor='#444',
                       linewidth=1.4, zorder=4)
    ax.add_patch(outer)

    # 直線分隔欄位
    for cx in col_x[1:]:
        ax.plot([cx, cx],
                [hdr_y - n_rows * row_h, hdr_y + row_h],
                color='#aaa', lw=0.8, zorder=4)

    plt.tight_layout()
    plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT}')


if __name__ == '__main__':
    main()
