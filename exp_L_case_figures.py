"""
EXP-L 報告書插圖製作
======================
圖A：成功辨識案例（Grad-CAM）
圖B：四張失敗案例對照 + 共通特徵分析
"""

import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

for fpath in (r'C:\Windows\Fonts\msjh.ttc', r'C:\Windows\Fonts\msyh.ttc'):
    if os.path.exists(fpath):
        fm.fontManager.addfont(fpath)
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

HM_DIR  = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L_chatgpt_image2\heatmaps')
OUT_A   = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L_chatgpt_image2\figL_success_case.png')
OUT_B   = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L_chatgpt_image2\figL_failed_cases.png')


# ──────────────────────────────────────────────────────────────────
# 圖A：成功辨識案例（拿 #1 山景湖泊作代表）
# ──────────────────────────────────────────────────────────────────
def fig_success():
    img = Image.open(HM_DIR / 'success_01.png').convert('RGB')
    fig = plt.figure(figsize=(15, 6))
    gs  = fig.add_gridspec(2, 1, height_ratios=[5, 1.2], hspace=0.15)
    ax_img = fig.add_subplot(gs[0])
    ax_txt = fig.add_subplot(gs[1])

    ax_img.imshow(img)
    ax_img.axis('off')

    cap = (
        '【判決】Fusion fake_prob = 99.15%（正確判為 AI 生成）\n'
        '【熱力圖觀察】CLIP 語義流（紅色集中於山峰輪廓與光影過渡帶）與 DIRE 重建誤差流'
        '（聚焦於前景反射與細節紋理）共同主導判決；FFT、DCT 兩個頻域流在天空、山體高頻處'
        '亦有顯著反應，反映多特徵互補有效運作。Noise SRM 殘差呈現均勻紋理分布，未顯示\n'
        '可疑訊號 — 與成功案例普遍特徵一致。'
    )
    ax_txt.axis('off')
    ax_txt.text(0.02, 0.95, cap, va='top', ha='left',
                fontsize=11.5, linespacing=1.6, color='#222',
                bbox=dict(facecolor='#e8f5e9', edgecolor='#3a7d44',
                          boxstyle='round,pad=0.6', lw=1.2))

    fig.suptitle('圖（X）EXP-L 成功辨識案例：山景湖泊（Image #1）— Per-Stream Grad-CAM 熱力圖',
                 fontsize=14, fontweight='bold', y=0.99)

    plt.savefig(OUT_A, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT_A}')


# ──────────────────────────────────────────────────────────────────
# 圖B：4 張失敗案例對照（垂直堆疊 + 共通解說）
# ──────────────────────────────────────────────────────────────────
def fig_failed():
    cases = [
        ('failed_15.png', '京都和服小巷夜景',  '混合場景', 3.89, -4.22),
        ('failed_21.png', '義大利小巷陽傘',    '混合場景', 1.22, -5.93),
        ('failed_23.png', '復古底片相機桌面',  '物件特寫', 1.20, -5.95),
        ('failed_25.png', '復古相機與配件',    '物件特寫', 3.68, -4.28),
    ]

    fig = plt.figure(figsize=(15, 14))
    gs  = fig.add_gridspec(5, 1, height_ratios=[3.0, 3.0, 3.0, 3.0, 1.6], hspace=0.12)

    for i, (fn, name, cat, prob, energy) in enumerate(cases):
        ax = fig.add_subplot(gs[i])
        img = Image.open(HM_DIR / fn).convert('RGB')
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(
            f'#{fn[7:9]} 【{cat}】{name}    '
            f'Fusion fake_prob = {prob}% (漏判)    Energy = {energy}',
            fontsize=11.5, fontweight='bold', color='#a13030', pad=6, loc='left')

    # 統一分析欄
    ax_txt = fig.add_subplot(gs[4])
    ax_txt.axis('off')
    cap = (
        '【共通特徵】四張漏判圖均屬「物件特寫」與「街景／建築」類別，具有：高度攝影風格化、'
        '電影感打光、復古色調、模擬真實鏡頭散景與光影過渡 — 紋理已高度逼近真實照片分布。\n'
        '【熱力圖共通觀察】CLIP 與 DIRE 兩個關鍵流的熱區呈現「分散且無清晰焦點」之模式，'
        '與成功案例「集中於主體輪廓」的熱圖結構顯著不同，反映模型未能定位有效判別特徵。\n'
        '【Energy 補救分析】四張漏判圖之 Energy 分數（-4.22 至 -5.95）反而高於成功案例'
        '（-7.04 至 -8.17），代表主模型不僅判錯，更「自信認為其屬於訓練分布內」，與'
        'EXP-J 之 ECE = 0.561 揭示之「不知道自己不知道」現象完全一致。\n'
        '【失敗原因推測】此類風格化內容在 OpenAI 之多模態訓練資料中可能比例極高，'
        '使 ChatGPT Image 2.0 對攝影風格化分布有極佳之模仿能力，造成本系統依賴之 CLIP'
        '語義特徵失去區辨力。'
    )
    ax_txt.text(0.02, 0.97, cap, va='top', ha='left',
                fontsize=11, linespacing=1.55, color='#222',
                bbox=dict(facecolor='#fde8e8', edgecolor='#a13030',
                          boxstyle='round,pad=0.6', lw=1.2))

    fig.suptitle('圖（X+1）EXP-L 辨識失敗案例：四張漏判圖之 Per-Stream Grad-CAM 對照',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(OUT_B, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT_B}')


if __name__ == '__main__':
    fig_success()
    fig_failed()
