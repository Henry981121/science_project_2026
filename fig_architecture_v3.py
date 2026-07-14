"""
科展研究架構圖 v3 — 依《全國科展初稿 v1》章節結構重繪
========================================================
- 階段一（準備階段）
- 階段二（實驗測試階段，三輪）
- 階段三（展示階段）
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

OUT = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig_architecture_v3.png')

# ── 色票（與報告書插圖風格一致）────────────────────────────────────
C_BG_S1      = '#d9e8f5'   # 階段一 淡藍
C_BD_S1      = '#5a8fbf'
C_BG_S2      = '#e8f5e9'   # 階段二 淡綠
C_BD_S2      = '#7cc77c'
C_BG_S3      = '#fff3cd'   # 階段三 淡黃
C_BD_S3      = '#d9a73a'
C_ROUND      = '#3a7d44'
C_BOX        = 'white'
C_BOX_BD     = '#555'
C_KEY        = '#cfe8d6'
C_KEY_BD     = '#3a7d44'
C_DERIV      = '#e8f0fa'   # 衍生分析（藍色系）
C_DERIV_BD   = '#4C72B0'
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


def round_tag(ax, x, y, text, w=3.0, fc=C_ROUND):
    h = 0.42
    bb = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.04",
                        linewidth=0, facecolor=fc, zorder=3)
    ax.add_patch(bb)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=4)


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.4, style='->'):
    ar = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, mutation_scale=13,
                          color=color, linewidth=lw, zorder=2)
    ax.add_patch(ar)


def container(ax, x, y, w, h, label, fc, ec):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.1",
                          linewidth=2, edgecolor=ec, facecolor=fc,
                          alpha=0.55, zorder=1)
    ax.add_patch(rect)
    tag_w, tag_h = 2.7, 0.5
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
    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 17)
    ax.axis('off')

    # ════════════════════════════════════════════════════════════════
    # 階段一（準備階段）— 左上
    # ════════════════════════════════════════════════════════════════
    container(ax, 0.3, 13.5, 17.5, 3.2,
              '階段一  準備階段', C_BG_S1, C_BD_S1)

    y = 14.7
    box(ax, 0.8, y, 2.5, 1.0,
        '撰寫核心程式碼\nsrc.zip 打包', fs=9)
    box(ax, 3.7, y, 2.2, 1.0, '環境設定', fs=9)
    box(ax, 6.3, y, 2.2, 1.0, '資料下載\n(192,053 張)', fs=9)
    box(ax, 8.9, y, 2.5, 1.0,
        '資料切分\nTrain / G1 / G2', fs=9)
    box(ax, 11.8, y - 0.1, 5.5, 1.15,
        '五流特徵提取設計（256×256→512 維 / 流）\n'
        'CLIP · FFT · DCT · DIRE · Noise',
        fs=9.5, fw='bold', fc=C_KEY, ec=C_KEY_BD)
    for x1, x2 in [(3.3, 3.7), (5.9, 6.3), (8.5, 8.9), (11.4, 11.8)]:
        arrow(ax, x1, y + 0.5, x2, y + 0.5)

    # 階段一 → 階段二 的縱向銜接
    arrow(ax, 14.5, 13.5, 14.5, 13.0, color=C_KEY_BD, lw=1.6)

    # ════════════════════════════════════════════════════════════════
    # 階段二（實驗測試階段）— 左中下
    # ════════════════════════════════════════════════════════════════
    container(ax, 0.3, 0.5, 17.5, 13.0,
              '階段二  實驗測試階段', C_BG_S2, C_BD_S2)

    # ── 第一輪：多視角融合驗證 ───────────────────────────────────────
    y = 11.6
    round_tag(ax, 0.7, y + 0.4, '第一輪  多視角融合驗證', w=3.9)
    box(ax, 0.7, y - 0.5, 2.4, 1.0,
        'EXP-A\n單一特徵效能矩陣\n(5 流獨立)', fs=9)
    box(ax, 3.4, y - 0.5, 2.4, 1.0,
        'EXP-B\n消融實驗\n(26 模型 + LOO)', fs=9)
    box(ax, 6.1, y - 0.5, 2.6, 1.0,
        'Shapley Value\n特徵流貢獻分析\n(EXP-B 延伸)',
        fs=8.5, fc=C_DERIV, ec=C_DERIV_BD)
    box(ax, 9.0, y - 0.5, 2.4, 1.0,
        'EXP-C\n融合策略比較\nConcat / W / CA', fs=9)
    box(ax, 11.7, y - 0.5, 5.4, 1.0,
        'MAIN 主模型完整訓練\n5 流 + Cross-Attention + GRL\nG1 ACC 98.43% / G2 ACC 90.01%',
        fs=9.5, fw='bold', fc=C_KEY, ec=C_KEY_BD)
    for x1, x2 in [(3.1, 3.4), (5.8, 6.1), (8.7, 9.0), (11.4, 11.7)]:
        arrow(ax, x1, y, x2, y)

    # ── 第二輪：性能測試與泛化 ───────────────────────────────────────
    y = 8.9
    round_tag(ax, 0.7, y + 0.4, '第二輪  性能測試與泛化', w=3.9)
    box(ax, 0.7, y - 0.5, 2.4, 1.0,
        'EXP-D\n泛化測試\n(G1 / G2 / G3)', fs=9)
    box(ax, 3.4, y - 0.5, 2.6, 1.0,
        'Silhouette + MMD\n特徵空間分佈量化',
        fs=8.5, fc=C_DERIV, ec=C_DERIV_BD)
    box(ax, 6.3, y - 0.5, 2.6, 1.0,
        'EXP-J\n錯誤類型分析\n(wfother FNR 57.6%)',
        fs=9, fc='#fde8e8', ec='#a13030')
    box(ax, 9.2, y - 0.5, 2.6, 1.0,
        'ECE\n信心校準分析\n(wfo ECE = 0.561)',
        fs=8.5, fc=C_DERIV, ec=C_DERIV_BD)
    for x1, x2 in [(3.1, 3.4), (6.0, 6.3), (8.9, 9.2)]:
        arrow(ax, x1, y, x2, y)

    # EXP-H 獨立驗真信號融合 — 第二輪的最後一個（橫跨右半，內含三子實驗）
    hi = FancyBboxPatch((12.0, y - 0.85), 5.2, 1.7,
                        boxstyle="round,pad=0.05",
                        linewidth=2, edgecolor='#e0a800',
                        facecolor='#fff2b0', alpha=0.6, zorder=1.5)
    ax.add_patch(hi)
    ax.text(14.6, y + 0.65,
            'EXP-H  獨立驗真信號融合〔核心突破〕',
            ha='center', fontsize=10, fontweight='bold',
            color='#c47f00', zorder=3)
    box(ax, 12.15, y - 0.55, 1.6, 0.95,
        'NSS 探索\n（飽和失效）',
        fs=8, fc='#fdf2f2', ec='#a13030')
    box(ax, 13.85, y - 0.55, 1.65, 0.95,
        'Triple Fusion\nMain+E+CA\nAUC +5.30',
        fs=8, fw='bold', fc='white', ec='#e0a800')
    box(ax, 15.6, y - 0.55, 1.55, 0.95,
        '階層覆寫\nS5 / S6\nFNR -39%',
        fs=8, fw='bold', fc='white', ec='#e0a800')
    arrow(ax, 11.8, y, 12.15, y - 0.08, color='#c47f00')

    # ── 第三輪：綜合比較與展望 ───────────────────────────────────────
    y = 6.0
    round_tag(ax, 0.7, y + 0.4, '第三輪  綜合比較與展望', w=3.9)
    box(ax, 0.7, y - 0.5, 2.4, 1.0,
        'EXP-F\nSOTA 比較\n(EffNet / CLIP)', fs=9)
    box(ax, 3.4, y - 0.5, 2.4, 1.0,
        'EXP-G\n穩健性測試\n(JPEG/Resize/Blur)', fs=9)
    box(ax, 6.1, y - 0.5, 2.4, 1.0,
        'mCE 總分\nmCE = 0.385\n(EXP-G 延伸)',
        fs=8.5, fc=C_DERIV, ec=C_DERIV_BD)
    box(ax, 8.8, y - 0.5, 2.4, 1.0,
        'EXP-E\n可解釋性分析\nGrad-CAM + Attn', fs=9)
    box(ax, 11.5, y - 0.5, 2.4, 1.0,
        'DAUC / IAUC\nGrad-CAM 忠實度\n(EXP-E 延伸)',
        fs=8.5, fc=C_DERIV, ec=C_DERIV_BD)
    box(ax, 14.2, y - 0.5, 2.9, 1.0,
        'EXP-I  多種子驗證\nMean 98.43% Std ±0.02%',
        fs=9, fc=C_KEY, ec=C_KEY_BD)
    for x1, x2 in [(3.1, 3.4), (5.8, 6.1), (8.5, 8.8), (11.2, 11.5),
                    (13.9, 14.2)]:
        arrow(ax, x1, y, x2, y)

    # ── 第四個區塊：組合特徵 AI 圖片辨識平台實作（屬於第三輪末段）──
    y = 3.6
    box(ax, 4.8, y - 0.55, 8.0, 1.1,
        '組合特徵 AI 圖片辨識平台實作  →  進入階段三\n'
        '(FastAPI 後端 + HTML 前端，封裝主模型 + Per-Stream 視覺化)',
        fs=10, fw='bold', fc=C_KEY, ec=C_KEY_BD)

    # ── 跨輪垂直引導 ────────────────────────────────────────────────
    arrow(ax, 14.4, 10.95, 14.4, 9.85, color=C_KEY_BD, lw=1.6)  # 一→二（主模型→泛化）
    arrow(ax, 1.85, 8.4, 1.85, 7.05, color='#888', lw=1.2)      # 二→三
    arrow(ax, 8.8, 5.45, 8.8, 4.2, color=C_KEY_BD, lw=1.6)      # 三→平台實作

    # ── 綜合評估結果（與原圖相同位置）──────────────────────────────
    box(ax, 4.8, 1.4, 8.0, 0.9,
        '綜合評估結果   G1 ACC 98.43%  /  G2 ACC 90.01%  /  Triple wfother AUC 90.07%',
        fs=10, fw='bold', fc=C_KEY, ec=C_KEY_BD)
    arrow(ax, 8.8, 3.05, 8.8, 2.3, color=C_KEY_BD, lw=1.6)

    # ════════════════════════════════════════════════════════════════
    # 階段三（展示階段）— 右側
    # ════════════════════════════════════════════════════════════════
    container(ax, 18.2, 0.5, 7.5, 16.2,
              '階段三  展示階段', C_BG_S3, C_BD_S3)

    sx, sw = 19.0, 6.0

    flow_items = [
        (15.0, 0.9, '推論輸入\n上傳圖片 / 即時拍照',
                                  C_BOX, C_BOX_BD, 10, 'normal'),
        (13.5, 1.0, '五流特徵提取管線\nCLIP · FFT · DCT · DIRE · Noise',
                                  C_BOX, C_BOX_BD, 9.5, 'normal'),
        (11.9, 1.0, 'Cross-Attention 融合推論\n(主模型 + GRL)',
                                  C_BOX, C_BOX_BD, 9.5, 'normal'),
        (10.0, 1.4, '推論輸出\n• 真偽判定結果\n• 各生成器來源機率\n• 信心分數',
                                  C_BOX, C_BOX_BD, 9.5, 'normal'),
        (7.4, 1.9, 'Grad-CAM 熱力圖視覺化\n（Per-Stream 視覺化）\n'
                   '• CLIP：Chefer relevance\n'
                   '• FFT/DCT/DIRE：Grad-CAM\n'
                   '• Noise：SRM residual',
                                  C_DERIV, C_DERIV_BD, 9, 'normal'),
        (4.4, 2.0, '組合特徵 AI 圖片辨識平台\n'
                   '（東區科展互動展示系統）\n'
                   '• FastAPI 後端\n'
                   '• HTML / JS 響應式前端\n'
                   '• 即時推論 + 5 流 XAI 顯示',
                                  C_KEY, C_KEY_BD, 9.5, 'bold'),
        (1.8, 1.5, '使用者體驗\n• 上傳即得判定結果\n'
                   '• 看懂 AI 為何如此判斷',
                                  '#fff9d9', '#e0a800', 9.5, 'bold'),
    ]

    prev_y = None
    for y0, h, txt, fc, ec, fs, fw in flow_items:
        box(ax, sx, y0, sw, h, txt, fc=fc, ec=ec, fs=fs, fw=fw)
        if prev_y is not None:
            arrow(ax, sx + sw / 2, prev_y, sx + sw / 2, y0 + h, lw=1.6)
        prev_y = y0

    # 綜合評估 → 階段三
    arrow(ax, 12.85, 1.85, sx + sw / 2, 15.0 + 0.45,
          lw=2, color=C_KEY_BD, style='->')

    # ════════════════════════════════════════════════════════════════
    # 圖例
    # ════════════════════════════════════════════════════════════════
    legend_x, legend_y = 0.5, 0.05
    items = [
        ('準備階段',         C_BG_S1,      C_BD_S1),
        ('實驗測試階段',     C_BG_S2,      C_BD_S2),
        ('展示階段',         C_BG_S3,      C_BD_S3),
        ('主實驗 (EXP-X)',   C_BOX,        C_BOX_BD),
        ('衍生分析',         C_DERIV,      C_DERIV_BD),
        ('關鍵成果',         C_KEY,        C_KEY_BD),
        ('核心突破 / 痛點',  '#fff2b0',    '#e0a800'),
    ]
    for i, (txt, fc, ec) in enumerate(items):
        x0 = legend_x + i * 2.55
        bb = FancyBboxPatch((x0, legend_y), 0.45, 0.28,
                            boxstyle="round,pad=0.03",
                            linewidth=1, edgecolor=ec, facecolor=fc)
        ax.add_patch(bb)
        ax.text(x0 + 0.55, legend_y + 0.14, txt, va='center', fontsize=9)

    # 主標題
    ax.text(13, 16.85, '科展研究架構圖 v3   —   基於組合特徵之 AI 生成影像鑑別技術探討',
            ha='center', fontsize=15, fontweight='bold')
    ax.text(13, 16.5,
            '《全國科展初稿 v1》章節結構：階段一 準備  →  階段二 三輪實驗  →  階段三 互動式展示',
            ha='center', fontsize=10.5, color='#444')

    plt.tight_layout()
    plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[saved] {OUT}")


if __name__ == '__main__':
    main()
