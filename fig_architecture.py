"""
科展研究架構圖 v2 — 重新編排（含 EXP-K、五大驗證、Per-Stream Grad-CAM、Web Demo）
==========================================================================
"""

import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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

OUT = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig_architecture_v2.png')

# ── 色票 ────────────────────────────────────────────────────────────
C_BG_STAGE2  = '#e8f5e9'
C_BD_STAGE2  = '#7cc77c'
C_BG_STAGE3  = '#fff3cd'
C_BD_STAGE3  = '#d9a73a'
C_ROUND      = '#3a7d44'
C_BOX        = 'white'
C_BOX_BD     = '#444'
C_HIGHLIGHT  = '#fff2b0'
C_HIGHLIGHT_BD = '#e0a800'
C_NEGATIVE   = '#f5e6e8'
C_NEGATIVE_BD = '#c4929e'
C_KEY        = '#cfe8d6'
C_KEY_BD     = '#3a7d44'
C_ARROW      = '#555'


def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_BOX_BD, fs=9, fw='normal',
        text_color='black', pad=0.05):
    bb = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad={pad}",
                        linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(bb)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, fontweight=fw, color=text_color,
            zorder=4, linespacing=1.4)


def round_tag(ax, x, y, text, w=2.5, fc=C_ROUND):
    h = 0.42
    bb = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.04",
                        linewidth=0, facecolor=fc, zorder=3)
    ax.add_patch(bb)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=4)


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, style='->'):
    ar = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, mutation_scale=14,
                          color=color, linewidth=lw, zorder=2)
    ax.add_patch(ar)


def container(ax, x, y, w, h, label, fc, ec):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.1",
                          linewidth=2, edgecolor=ec, facecolor=fc,
                          alpha=0.55, zorder=1)
    ax.add_patch(rect)
    tag_w, tag_h = 2.5, 0.5
    tx, ty = x + 0.3, y + h - 0.6
    rb = FancyBboxPatch((tx, ty), tag_w, tag_h,
                        boxstyle="round,pad=0.04",
                        linewidth=1.5, edgecolor=ec,
                        facecolor='white', zorder=2)
    ax.add_patch(rb)
    ax.text(tx + tag_w / 2, ty + tag_h / 2, label,
            ha='center', va='center',
            fontsize=11, fontweight='bold', color=ec, zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(22, 15))
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 18)
    ax.axis('off')

    # ═══════════════════════════════════════════════════════════════
    # 階段二
    # ═══════════════════════════════════════════════════════════════
    container(ax, 0.3, 0.5, 17.5, 17, '階段二  實驗測試階段',
              C_BG_STAGE2, C_BD_STAGE2)

    # ── 第一輪 ─────────────────────────────────────────────────────
    y = 15.3
    round_tag(ax, 0.7, y + 0.4, '第一輪  模型建構', w=3.4)
    box(ax, 0.7, y - 0.5, 2.3, 1.0, 'EXP-A\n單一特徵效能矩陣\n(5 流)', fs=9)
    box(ax, 3.5, y - 0.5, 2.3, 1.0, 'EXP-B\n消融實驗 LOO', fs=9)
    box(ax, 6.3, y - 0.5, 2.3, 1.0, 'EXP-C\n融合策略比較\n(Concat/W/CA)', fs=9)
    box(ax, 9.4, y - 0.5, 3.5, 1.0,
        'MAIN 主模型完整訓練\n5 流 GRL + Cross-Attention\nVal AUC 99.85%',
        fs=9.5, fw='bold', fc=C_KEY, ec=C_KEY_BD)
    arrow(ax, 3.0, y, 3.5, y); arrow(ax, 5.8, y, 6.3, y)
    arrow(ax, 8.6, y, 9.4, y)

    # ── 第二輪 ─────────────────────────────────────────────────────
    y = 13.5
    round_tag(ax, 0.7, y + 0.4, '第二輪  泛化與失敗診斷', w=3.9)
    box(ax, 0.7, y - 0.5, 2.3, 1.0, 'EXP-D\n泛化測試\n(G1 / G2 / G3)', fs=9)
    box(ax, 3.5, y - 0.5, 2.3, 1.0, 'EXP-D2\nt-SNE 視覺化', fs=9)
    box(ax, 6.3, y - 0.5, 3.5, 1.0,
        'EXP-J  錯誤類型分析\nwfother FNR 57.6%（核心痛點）',
        fs=9, fc=C_NEGATIVE, ec=C_NEGATIVE_BD)
    arrow(ax, 3.0, y, 3.5, y); arrow(ax, 5.8, y, 6.3, y)

    # ── 第三輪 NSS 負面 ──────────────────────────────────────────
    y = 11.6
    round_tag(ax, 0.7, y + 0.4, '第三輪  NSS 探索（負面結果）', w=4.5)
    box(ax, 0.7, y - 0.7, 4.5, 1.2,
        'EXP-H NSS 系列實驗\n• v1：NSS concat (5 流 + 36)\n• v2：NSS 投影 6th stream\n• v3：wildfake_other 專項',
        fs=8.5, fc=C_NEGATIVE, ec=C_NEGATIVE_BD)
    box(ax, 5.7, y - 0.7, 4.5, 1.2,
        'NSS 2×2 設計（Variants A-D）\n• A：NSS 36 融合 / B：NSS 36 獨立 Mahalanobis\n• C：NSS++ 79 融合 / D：NSS++ 79 獨立',
        fs=8.5, fc=C_NEGATIVE, ec=C_NEGATIVE_BD)
    box(ax, 10.7, y - 0.45, 6.2, 0.7,
        '結論：NSS 信號飽和，2024+ Diffusion 已能模仿低階統計',
        fs=9, fw='bold', text_color='#a13030',
        fc='#fdf2f2', ec='#a13030')
    arrow(ax, 5.2, y - 0.1, 5.7, y - 0.1)
    arrow(ax, 10.2, y - 0.1, 10.7, y - 0.1)

    # ── 第四輪 EXP-K 核心突破 ──────────────────────────────────────
    y = 9.5
    hi = FancyBboxPatch((0.5, y - 0.95), 16.8, 1.7,
                        boxstyle="round,pad=0.05",
                        linewidth=2.5, edgecolor=C_HIGHLIGHT_BD,
                        facecolor=C_HIGHLIGHT, alpha=0.55, zorder=1.5)
    ax.add_patch(hi)
    round_tag(ax, 0.7, y + 0.4, '第四輪  獨立驗真信號融合〔核心突破〕',
              w=5.6, fc='#c47f00')
    box(ax, 0.8, y - 0.75, 3.4, 1.2,
        'EXP-K1\n單信號 α 掃描\n(CA vs Energy)\nEnergy α=0.05 +3.14 AUC',
        fs=8.5)
    box(ax, 4.6, y - 0.75, 4.0, 1.2,
        'EXP-K2\nTriple-Signal Late Fusion\nS = S_main + 0.05·E + 0.05·CA\nwfother AUC +5.30',
        fs=9, fw='bold', fc='#fff9d9', ec=C_HIGHLIGHT_BD)
    box(ax, 9.0, y - 0.75, 4.0, 1.2,
        'EXP-K3\n階層式覆寫策略\nS5 Tiered：FNR 57.6→18.6%\nS6 Vote：AUC 89.95',
        fs=9, fw='bold', fc='#fff9d9', ec=C_HIGHLIGHT_BD)
    box(ax, 13.4, y - 0.45, 3.6, 0.7,
        '成果：wfother FNR\n下降 38.98 個百分點',
        fs=9, fw='bold', text_color='#1f7a1f',
        fc='#e8f8e8', ec='#1f7a1f')
    arrow(ax, 4.2, y - 0.15, 4.6, y - 0.15, lw=1.8)
    arrow(ax, 8.6, y - 0.15, 9.0, y - 0.15, lw=1.8)
    arrow(ax, 13.0, y - 0.15, 13.4, y - 0.15, lw=1.8)

    # ── 第五輪 ─────────────────────────────────────────────────────
    y = 7.0
    round_tag(ax, 0.7, y + 0.4, '第五輪  SOTA 比較與穩健性', w=4.2)
    box(ax, 0.7, y - 0.5, 2.6, 1.0,
        'EXP-F\nSOTA 比較\n(ResNet50/EffNet/CLIP)', fs=9)
    box(ax, 3.8, y - 0.5, 2.6, 1.0,
        'EXP-G\n穩健性測試\n13 種劣化', fs=9)
    box(ax, 6.9, y - 0.5, 2.6, 1.0,
        'EXP-I\n多種子穩定性\nStd ± 0.02%', fs=9)
    arrow(ax, 3.3, y, 3.8, y); arrow(ax, 6.4, y, 6.9, y)

    # ── 第六輪 可解釋性 + 五大驗證 ──────────────────────────────────
    y = 4.3
    round_tag(ax, 0.7, y + 1.4, '第六輪  可解釋性與驗證方法', w=4.2)

    box(ax, 0.7, y + 0.4, 4.0, 1.05,
        'EXP-E 可解釋性分析\n• Per-Stream Grad-CAM\n• Chefer CLIP relevance',
        fs=9)
    box(ax, 4.9, y + 0.4, 4.2, 1.05,
        'Level 2  注意力 + Ablation\n• Noise Decisive D2 = 8.37%\n• Cross-Attention 權重分析',
        fs=9)

    # 五大驗證方法 — header + 5 box
    box(ax, 9.3, y + 1.15, 7.7, 0.5,
        '五大驗證方法（V1 - V5）', fs=10, fw='bold',
        fc=C_KEY, ec=C_KEY_BD)
    methods = [
        ('V1\nDAUC/IAUC\n(忠實度)',  '#e8f0fa'),
        ('V2\nShapley\n(貢獻度)',     '#e8f0fa'),
        ('V3\nmCE\n(劣化魯棒)',       '#e8f0fa'),
        ('V4\nECE\n(校準度)',         '#e8f0fa'),
        ('V5\nSilh.+MMD\n(分布)',     '#e8f0fa'),
    ]
    bw, gap = 1.45, 0.1
    bx0 = 9.4
    for i, (txt, c) in enumerate(methods):
        box(ax, bx0 + i * (bw + gap), y + 0.4, bw, 0.7, txt,
            fs=8.5, fc=c, ec='#4C72B0')

    # ── 跨輪垂直主軸引導 ────────────────────────────────────────────
    # 把主軸畫在 x=1.85，從 MAIN(第一輪) 順流到第六輪
    arrow(ax, 11.15, 14.8, 11.15, 14.0, color='#3a7d44', lw=1.6)  # MAIN→第二輪
    arrow(ax, 1.85, 13.0, 1.85, 12.4, color='#888', lw=1.2)  # 二→三
    arrow(ax, 1.85, 10.9, 1.85, 10.25, color='#888', lw=1.2)  # 三→四
    arrow(ax, 1.85, 8.55, 1.85, 7.5, color='#888', lw=1.2)   # 四→五
    arrow(ax, 1.85, 6.5, 1.85, 5.75, color='#888', lw=1.2)   # 五→六

    # ── 綜合評估結果 ──────────────────────────────────────────────
    box(ax, 6.3, 1.4, 5.0, 1.05,
        '綜合評估結果\nMain G1 ACC 98.54%  /  Triple wfother AUC 90.07%',
        fs=10, fw='bold', fc=C_KEY, ec=C_KEY_BD)
    # 第六輪 → 綜合評估
    arrow(ax, 8.8, 4.3, 8.8, 2.45, color='#3a7d44', lw=1.6)

    # ═══════════════════════════════════════════════════════════════
    # 階段三 展示階段
    # ═══════════════════════════════════════════════════════════════
    container(ax, 18.2, 0.5, 7.5, 17, '階段三  展示階段',
              C_BG_STAGE3, C_BD_STAGE3)

    sx = 19.0
    sw = 6.0

    flow_items = [
        (15.5, 0.85, '推論輸入\n上傳圖片 / 即時拍照',
                              C_BOX, C_BOX_BD, 10, 'normal'),
        (14.0, 0.95, '5 流特徵提取管線\nCLIP · FFT · DCT · DIRE · Noise',
                              C_BOX, C_BOX_BD, 9.5, 'normal'),
        (12.4, 0.95, 'Main 融合推論\n(Cross-Attention + GRL)',
                              C_BOX, C_BOX_BD, 9.5, 'normal'),
        (10.7, 1.15, '獨立驗真信號計算〔核心〕\n+ Energy Score\n+ Chromatic Aberration',
                              '#fff9d9', C_HIGHLIGHT_BD, 9.5, 'bold'),
        (8.85, 1.15, 'Triple Fusion / 階層覆寫〔核心〕\nS5 Tiered（FNR -39%）\nS6 Vote（AUC 89.95）',
                              '#fff9d9', C_HIGHLIGHT_BD, 9.5, 'bold'),
        (7.05, 1.05, '推論輸出\n真假判斷 + 來源機率 + 信心度',
                              C_BOX, C_BOX_BD, 9.5, 'normal'),
        (5.0, 1.5, 'Per-Stream Grad-CAM 視覺化\n• CLIP：Chefer relevance\n• FFT/DCT/DIRE：Grad-CAM\n• Noise：SRM residual',
                              '#e8f0fa', '#4C72B0', 9, 'normal'),
        (2.65, 1.6, 'Web Demo（east_zone_project）\n• FastAPI 後端\n• HTML / JS 響應式前端\n• 即時推論 + 5 流 XAI 顯示',
                              C_KEY, C_KEY_BD, 9.5, 'bold'),
    ]

    prev_y = None
    for y0, h, txt, fc, ec, fs, fw in flow_items:
        box(ax, sx, y0, sw, h, txt, fc=fc, ec=ec, fs=fs, fw=fw)
        if prev_y is not None:
            arrow(ax, sx + sw / 2, prev_y, sx + sw / 2, y0 + h, lw=1.6)
        prev_y = y0

    # 綜合評估 → 階段三（曲線箭頭）
    arrow(ax, 11.3, 1.9, sx + sw / 2, 15.5 + 0.45,
          lw=2, color='#3a7d44', style='->')

    # ═══════════════════════════════════════════════════════════════
    # 圖例
    # ═══════════════════════════════════════════════════════════════
    legend_x, legend_y = 0.5, 0.05
    items = [
        ('一般實驗',         C_BOX,       C_BOX_BD),
        ('核心突破（黃）',   C_HIGHLIGHT, C_HIGHLIGHT_BD),
        ('負面結果（粉）',   C_NEGATIVE,  C_NEGATIVE_BD),
        ('關鍵成果（綠）',   C_KEY,       C_KEY_BD),
        ('驗證方法（藍）',   '#e8f0fa',   '#4C72B0'),
    ]
    for i, (txt, fc, ec) in enumerate(items):
        x0 = legend_x + i * 3.5
        bb = FancyBboxPatch((x0, legend_y), 0.45, 0.28,
                            boxstyle="round,pad=0.03",
                            linewidth=1, edgecolor=ec, facecolor=fc)
        ax.add_patch(bb)
        ax.text(x0 + 0.6, legend_y + 0.14, txt, va='center', fontsize=9.5)

    # 主標題
    ax.text(13, 17.6, '科展研究總架構圖 v2  —  AI 生成影像偵測系統',
            ha='center', fontsize=15, fontweight='bold')
    ax.text(13, 17.25,
            '5 流融合主模型  +  EXP-K 獨立驗真信號融合（核心突破）  +  五大驗證方法  +  互動式 Web Demo',
            ha='center', fontsize=10.5, color='#444')

    plt.tight_layout()
    plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[saved] {OUT}")


if __name__ == '__main__':
    main()
