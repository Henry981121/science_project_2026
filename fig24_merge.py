"""
合併 fig24_a_alpha_sweep.png + fig24_b_triple_fusion.png 為單一圖檔，
並另存 panel c 為獨立檔（改名以符合科展編號）。
"""

import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r'C:\Users\harry\OneDrive\Desktop\outputs')
SRC_A = ROOT / 'fig24_a_alpha_sweep.png'
SRC_B = ROOT / 'fig24_b_triple_fusion.png'
SRC_C = ROOT / 'fig24_c_pareto.png'

OUT_AB = ROOT / 'fig24_late_fusion_analysis.png'
OUT_C  = ROOT / 'fig25_override_pareto_analysis.png'


def load_cjk_font(size: int):
    for fp in (r'C:\Windows\Fonts\msjh.ttc',
               r'C:\Windows\Fonts\msyh.ttc',
               r'C:\Windows\Fonts\mingliu.ttc'):
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def merge_ab():
    img_a = Image.open(SRC_A).convert('RGB')
    img_b = Image.open(SRC_B).convert('RGB')

    # 統一高度（取較大者）
    h = max(img_a.height, img_b.height)
    if img_a.height != h:
        img_a = img_a.resize((int(img_a.width * h / img_a.height), h), Image.LANCZOS)
    if img_b.height != h:
        img_b = img_b.resize((int(img_b.width * h / img_b.height), h), Image.LANCZOS)

    gap         = 40        # 中間留白
    title_h     = 120       # 上方總標題區
    margin      = 30        # 上下左右邊距

    W = img_a.width + img_b.width + gap + margin * 2
    H = h + title_h + margin * 2

    canvas = Image.new('RGB', (W, H), 'white')
    canvas.paste(img_a, (margin, margin + title_h))
    canvas.paste(img_b, (margin + img_a.width + gap, margin + title_h))

    # 主標題 + 副標題
    draw = ImageDraw.Draw(canvas)
    font_title = load_cjk_font(36)
    font_sub   = load_cjk_font(22)

    title = '圖（二十四）獨立驗真信號之 Late Fusion 分析：' \
            'α 掃描與三信號融合 AUC 比較'
    sub   = '（左：CA 與 Energy 之單獨融合 α 掃描曲線；右：' \
            'S_main + 0.05·S_energy + 0.05·S_CA 三信號融合於 G1 / G2 / wfother 之 AUC）'

    # 標題置中
    tb = draw.textbbox((0, 0), title, font=font_title)
    tw = tb[2] - tb[0]
    draw.text(((W - tw) / 2, margin + 10), title,
              fill='black', font=font_title)

    sb = draw.textbbox((0, 0), sub, font=font_sub)
    sw = sb[2] - sb[0]
    draw.text(((W - sw) / 2, margin + 65), sub,
              fill='#444', font=font_sub)

    canvas.save(OUT_AB, dpi=(200, 200))
    print(f'[saved] {OUT_AB}  ({W}x{H})')


def rename_c():
    """把 panel c 加上新標題另存為 fig25。"""
    img_c = Image.open(SRC_C).convert('RGB')
    margin = 30
    title_h = 120
    W = img_c.width + margin * 2
    H = img_c.height + title_h + margin * 2

    canvas = Image.new('RGB', (W, H), 'white')
    canvas.paste(img_c, (margin, margin + title_h))

    draw = ImageDraw.Draw(canvas)
    font_title = load_cjk_font(36)
    font_sub   = load_cjk_font(22)

    title = '圖（二十五）獨立驗真信號之覆寫策略：' \
            '於 wildfake_other 之 FNR–FPR 權衡 Pareto 分析'
    sub   = '（標註 S5 Tiered 與 S6 Vote 兩個關鍵策略點；' \
            '左下黃星為理想 FNR=FPR=0 之參考點）'

    tb = draw.textbbox((0, 0), title, font=font_title)
    tw = tb[2] - tb[0]
    draw.text(((W - tw) / 2, margin + 10), title,
              fill='black', font=font_title)

    sb = draw.textbbox((0, 0), sub, font=font_sub)
    sw = sb[2] - sb[0]
    draw.text(((W - sw) / 2, margin + 65), sub,
              fill='#444', font=font_sub)

    canvas.save(OUT_C, dpi=(200, 200))
    print(f'[saved] {OUT_C}  ({W}x{H})')


if __name__ == '__main__':
    merge_ab()
    rename_c()
