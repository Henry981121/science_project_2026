"""
圖（二十六）覆寫策略邏輯圖解 — S0 至 S6 的決策流程
========================================================
為圖（二十五）的 Pareto 散點圖補上「每個策略到底在做什麼」的說明，
讓讀者能對應 Pareto 上的位置理解每種策略的設計理念。
"""

import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

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

OUT = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig26_strategy_explain.png')


# ──────────────────────────────────────────────────────────────────
# 共用：策略卡片繪製
# ──────────────────────────────────────────────────────────────────
def card(ax, x, y, w, h, *, header_color, marker_shape, marker_color,
         strategy_id, strategy_name, formula, behavior, pareto_pos,
         risk_level, use_case):
    """
    在 ax 上畫一個策略卡片。
    """
    # 卡片外框
    outer = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.04",
                            linewidth=1.5, edgecolor='#666',
                            facecolor='white', zorder=2)
    ax.add_patch(outer)

    # 頂部色條（含策略 ID + marker）
    hdr_h = 0.55
    hdr = FancyBboxPatch((x, y + h - hdr_h), w, hdr_h,
                          boxstyle="round,pad=0.04",
                          linewidth=0, facecolor=header_color, zorder=3)
    ax.add_patch(hdr)

    # 左側 marker
    ax.scatter([x + 0.32], [y + h - hdr_h / 2],
               marker=marker_shape, s=260,
               color=marker_color, edgecolor='black',
               linewidth=1.2, zorder=4)
    # 策略 ID + 名稱
    ax.text(x + 0.62, y + h - hdr_h / 2,
            f'{strategy_id}  —  {strategy_name}',
            va='center', ha='left',
            fontsize=11, fontweight='bold', color='white', zorder=4)

    # 決策公式區塊
    body_y = y + h - hdr_h - 0.1
    ax.text(x + 0.15, body_y - 0.05, '判決規則：',
            va='top', ha='left', fontsize=9, color='#666', zorder=4)

    # 公式 box（淡灰底）
    fb = FancyBboxPatch((x + 0.15, body_y - 0.78), w - 0.3, 0.7,
                         boxstyle="round,pad=0.03",
                         linewidth=0.8, edgecolor='#bbb',
                         facecolor='#f5f5f5', zorder=3)
    ax.add_patch(fb)
    # 不使用 monospace（內含中文，monospace 字體沒 CJK 會變方塊）
    ax.text(x + w / 2, body_y - 0.43, formula,
            va='center', ha='center',
            fontsize=8.8, color='#222',
            linespacing=1.4, zorder=4)

    # 行為描述
    ax.text(x + 0.15, body_y - 0.92, '行為：',
            va='top', ha='left', fontsize=9, color='#666', zorder=4)
    ax.text(x + 0.15, body_y - 1.10, behavior,
            va='top', ha='left', fontsize=8.8, color='#222',
            wrap=True, zorder=4)

    # 底部三欄資訊：Pareto 位置 / 風險 / 適用場景
    info_y = y + 0.18
    ax.text(x + 0.15, info_y + 0.5, f'⊙ Pareto 位置：{pareto_pos}',
            va='top', fontsize=8.3, color='#444', zorder=4)
    ax.text(x + 0.15, info_y + 0.27, f'⊙ 風險：{risk_level}',
            va='top', fontsize=8.3, color='#444', zorder=4)
    ax.text(x + 0.15, info_y + 0.04, f'⊙ 適用：{use_case}',
            va='top', fontsize=8.3, color='#444', zorder=4)


# ──────────────────────────────────────────────────────────────────
# 主要繪圖
# ──────────────────────────────────────────────────────────────────
def main():
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 17)
    ax.axis('off')

    # ── 標題 ────────────────────────────────────────────────────
    ax.text(12, 16.4,
            '圖（二十六）覆寫策略邏輯圖解 — S0 至 S6 的決策流程',
            ha='center', fontsize=16, fontweight='bold')
    ax.text(12, 16.0,
            '對應圖（二十五）Pareto 上的每個策略點，說明其決策規則與設計理念',
            ha='center', fontsize=11, color='#555')

    # ── 三個輸入信號示意（最上方）──────────────────────────────────
    ax.text(12, 15.2, '三個輸入信號',
            ha='center', fontsize=12, fontweight='bold', color='#3a7d44')

    signals = [
        (4.5,  '#999999', 'S_main',     '主模型 fake 機率\n(5 流 GRL Cross-Attn)',
         '訓練分布內準確；對未知生成器有過度自信現象'),
        (12.0, '#4C72B0', 'S_energy',   '能量分數 (Energy Score)\nT·logsumexp(logits/T)',
         '對 OOD 樣本敏感；可揭露主模型的「不熟悉」'),
        (19.5, '#55A868', 'S_CA',       '色像差信號 (Chromatic Aberration)\nRGB 三通道亞像素徑向偏移',
         '與訓練分布完全獨立的物理光學痕跡；單獨弱、組合強'),
    ]
    for cx, c, name, desc, role in signals:
        bb = FancyBboxPatch((cx - 1.8, 13.4), 3.6, 1.6,
                            boxstyle="round,pad=0.05",
                            linewidth=1.5, edgecolor=c, facecolor='white', zorder=3)
        ax.add_patch(bb)
        ax.text(cx, 14.75, name, ha='center', fontsize=11,
                fontweight='bold', color=c, zorder=4)
        ax.text(cx, 14.35, desc, ha='center', fontsize=8.5,
                color='#222', zorder=4)
        ax.text(cx, 13.78, role, ha='center', fontsize=8,
                color='#666', style='italic', zorder=4, wrap=True)

    # 從信號到策略區的引導
    for cx in (4.5, 12.0, 19.5):
        ar = FancyArrowPatch((cx, 13.4), (cx, 12.95),
                              arrowstyle='->', mutation_scale=14,
                              color='#888', lw=1.4)
        ax.add_patch(ar)

    # ── 7 個策略卡片（3 列 × 3 行，最後一格留作圖例）────────────
    card_w, card_h = 7.2, 3.8
    cols_x = [0.4, 8.4, 16.4]
    rows_y = [8.6, 4.4, 0.2]

    strategies = [
        # (header_color, marker, strategy_id, name, formula, behavior, pareto, risk, use_case)
        ('#222222', 'o', '#222',
         'S0',  'Baseline （基線）',
         'pred = (S_main > 0.5)',
         '只用主模型輸出，未啟用任何獨立信號。',
         '左上角 (FPR約1.5%, FNR約57.6%)',
         '對未知生成器嚴重漏判',
         '比較基準'),

        ('#C44E52', 's', '#C44E52',
         'S1',  'Energy Override （Energy 覆寫）',
         'if S_energy > τ:  pred = AI\nelse:           pred = (S_main > 0.5)',
         'Energy 一旦超過閾值就強制判 AI，不再參考主模型。\n'
         '掃描 τ 取 {0.5, 0.6, 0.7, 0.8} 共 4 點。',
         '中間段（隨 τ 變化）',
         'τ 太低 → FPR 暴增；τ 太高 → 效果有限',
         '單信號嘗試'),

        ('#55A868', '^', '#55A868',
         'S2',  'CA Override （色像差覆寫）',
         'if S_CA > τ:  pred = AI\nelse:         pred = (S_main > 0.5)',
         '色像差過閾值就強制判 AI。\n'
         '掃描 τ 取 {0.5, 0.6, 0.7} 共 3 點。',
         '右上至右中（CA 單獨偏弱）',
         '單獨用 CA → FPR 高、效益低',
         '驗證 CA 不能單獨用'),

        ('#8172B2', 'D', '#8172B2',
         'S3',  'OR （任一觸發即覆寫）',
         'if (S_energy > τE) OR (S_CA > τC):\n    pred = AI\nelse: pred = (S_main > 0.5)',
         '兩個獨立信號任一過閾值就判 AI 。\n'
         '掃描 (τE, τC) 共 3 組。',
         '右下（FNR 極低但 FPR 極高）',
         '最激進；幾乎抓到所有 AI，但誤殺真照片',
         '抓取率優先 / 不重視 FPR'),

        ('#CCB974', 'v', '#CCB974',
         'S4',  'AND （雙信號同意才覆寫）',
         'if (S_energy > τE) AND (S_CA > τC):\n    pred = AI\nelse: pred = (S_main > 0.5)',
         '兩個獨立信號都必須過閾值才判 AI 。\n'
         '掃描 (τE, τC) 共 2 組。',
         '左中（FPR 較低但 FNR 偏高）',
         '最保守；不易誤殺但抓不到太多 AI',
         '避免誤判優先'),

        ('#4C72B0', 'P', '#4C72B0',
         'S5 [ * ]', 'Tiered （階層式覆寫）',
         'if S_energy > 0.7:                       # 高自信\n'
         '    pred = AI\nelif (S_energy > 0.5) AND (S_CA > 0.6):  # 聯合\n'
         '    pred = AI\nelse: pred = (S_main > 0.5)',
         '兩階層：先用 Energy 高閾值快速判 AI；\n'
         '未觸發者再用 Energy+CA 較寬閾值聯合覆寫。',
         '中段 (FPR=21.6%, FNR=18.6%) [ * ]',
         'FNR 大降至 18.6% 但 FPR 升至 21.6%',
         '高敏感場景（媒體驗證、新聞查核）'),

        ('#FF8C00', '*', '#FF8C00',
         'S6 [ * ]', 'Vote （三信號軟投票）',
         'S_vote = wm·S_main + we·S_energy + wc·S_CA\npred = (S_vote > 0.5)',
         '三信號加權平均後再決策（不採覆寫）。\n'
         '預設權重 (wm, we, wc) = (0.5, 0.25, 0.25)。',
         '左上 (FPR=1.6%, FNR=56.4%) [ * ]',
         'FNR 幾乎與 baseline 同；但 AUC 提升至 89.95',
         '低誤判場景（一般用戶端工具）'),
    ]

    # 排列：3×3 grid（前 7 格放策略，第 8 格放圖例）
    positions = []
    for row in rows_y:
        for col in cols_x:
            positions.append((col, row))

    for i, (st, pos) in enumerate(zip(strategies, positions)):
        x, y = pos
        (hdr_c, mk, mk_c, sid, name, formula, behavior,
         pareto, risk, use_case) = st
        card(ax, x, y, card_w, card_h,
             header_color=hdr_c, marker_shape=mk, marker_color=mk_c,
             strategy_id=sid, strategy_name=name,
             formula=formula, behavior=behavior,
             pareto_pos=pareto, risk_level=risk, use_case=use_case)

    # ── 第 8 格：整體說明 ─────────────────────────────────────
    x, y = positions[7]
    legend = FancyBboxPatch((x, y), card_w, card_h,
                             boxstyle="round,pad=0.04",
                             linewidth=1.5, edgecolor='#3a7d44',
                             facecolor='#e8f8e8', zorder=2)
    ax.add_patch(legend)
    ax.text(x + card_w / 2, y + card_h - 0.4,
            '如何選擇策略？',
            ha='center', fontsize=12, fontweight='bold', color='#1f7a1f')

    explain_lines = [
        '• 覆寫策略 (S1–S5)：',
        '   當獨立信號夠強時直接「強制改判」',
        '   主模型輸出 → 不採用',
        '',
        '• 軟投票策略 (S6)：',
        '   加權平均三信號後再判決',
        '   主模型輸出 → 部分採用',
        '',
        '• Pareto 上沒有「全勝」策略，',
        '  只有依場景需求的最佳折衷：',
        '   •高敏感場景 → S5 Tiered [ * ]',
        '   •低誤判場景 → S6 Vote [ * ]',
    ]
    for i, line in enumerate(explain_lines):
        ax.text(x + 0.25, y + card_h - 0.95 - i * 0.23,
                line, ha='left', fontsize=9, color='#222')

    plt.tight_layout()
    plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT}')


if __name__ == '__main__':
    main()
