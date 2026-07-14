"""
EXP-L：25 張 ChatGPT Image 2.0 縮圖總覽
========================================
每張圖標上索引 + 主模型判決 + Energy 分數，方便用戶分類。
"""

import os, csv
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

IMG_DIR = Path(r'C:\Users\harry\OneDrive\Desktop\CHAT-GPT_IMAGE_2.0')
CSV_PATH = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L_chatgpt_image2\per_image_results.csv')
OUT = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L_chatgpt_image2\contact_sheet.png')


def main():
    # 讀預測結果
    pred = {}
    with open(CSV_PATH, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            pred[int(row['idx'])] = row

    images = sorted([p for p in IMG_DIR.iterdir()
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')])

    n = len(images)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.6))

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i >= n:
            ax.axis('off')
            continue
        idx = i + 1
        img = Image.open(images[i]).convert('RGB')
        img.thumbnail((512, 512))
        ax.imshow(img)
        ax.axis('off')

        # 判決資訊
        r = pred.get(idx, {})
        prob = float(r.get('fusion_prob_fake', 0))
        is_correct = r.get('is_correct', '').lower() == 'true'
        symbol = '[O]' if is_correct else '[X]'
        color  = '#1f7a1f' if is_correct else '#a13030'
        title = f'#{idx:02d}  {symbol}  fusion {prob:.1f}%'
        ax.set_title(title, fontsize=11, fontweight='bold', color=color, pad=4)

    fig.suptitle('EXP-L  ChatGPT Image 2.0  —  25 張測試圖總覽\n'
                 '[O] 正確判為 AI（21 張）       [X] 漏判為真實（4 張）',
                 fontsize=15, fontweight='bold', y=0.997)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT}')


if __name__ == '__main__':
    main()
