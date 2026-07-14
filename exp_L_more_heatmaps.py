"""
EXP-L 補跑：更多成功案例 Grad-CAM，挑 CLIP 熱區清晰的圖
=========================================================
候選：#2 貓、#5 向日葵、#7 狗、#8 棕櫚樹、#11 巴黎鐵塔、#19 山景城堡
"""

import os, sys, time
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_DIR = Path(r'C:\Users\harry\Downloads\east_zone_project_v2\east_zone_project')
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, r'C:\Users\harry\OneDrive\Desktop\science_project_2026_v2')

from detector import AIDetector, STREAMS, EVAL_TF, DEVICE

IMG_DIR = Path(r'C:\Users\harry\OneDrive\Desktop\CHAT-GPT_IMAGE_2.0')
HM_DIR  = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L_chatgpt_image2\heatmaps')
HM_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = [2, 5, 7, 8, 11, 14, 17, 19]   # 主體明顯的候選


def main():
    images = sorted([p for p in IMG_DIR.iterdir()
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')])

    print(f'[load] Loading AIDetector...')
    t0 = time.time()
    det = AIDetector()
    print(f'[load] Done {time.time()-t0:.1f}s')
    explainer = det.explainer

    for idx in CANDIDATES:
        if idx > len(images):
            continue
        img_path = images[idx - 1]
        img_pil = Image.open(img_path).convert('RGB')
        img_224 = img_pil.resize((224, 224))
        img_np = np.array(img_224)
        img_tensor = EVAL_TF(img_pil).unsqueeze(0).to(DEVICE)

        # 主模型 fake_prob
        with torch.no_grad():
            feats = [det.extractors[s].extract_features(img_tensor) for s in STREAMS]
            fused = torch.cat(feats, dim=1)
            lb, _, _, _ = det.fusion(fused, grl_lambda=0)
            prob = F.softmax(lb / 3.0, dim=1)[0, 1].item() * 100

        explanations = explainer.explain_image(img_tensor)
        save_path = HM_DIR / f'success_{idx:02d}_pick.png'
        explainer.visualize(
            img_np, explanations,
            save_path=str(save_path),
            title=f'EXP-L Image #{idx} (success)  | fusion fake_prob (T=3) = {prob:.2f}%'
        )
        print(f'  [#{idx:2d}] fusion={prob:.2f}%  →  {save_path.name}')


if __name__ == '__main__':
    main()
