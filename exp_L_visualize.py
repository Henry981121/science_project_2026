"""
EXP-L 視覺化與 Energy 補救分析
==================================
1. 對 4 張漏判 + 3 張成功對照圖產生 Per-Stream Grad-CAM 1x6 對照圖
2. 計算每張圖的 Energy 分數，看 +Energy 融合能否救回漏判
"""

import os, sys, json
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_DIR = Path(r'C:\Users\harry\Downloads\east_zone_project_v2\east_zone_project')
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, r'C:\Users\harry\OneDrive\Desktop\science_project_2026_v2')

from detector import AIDetector, STREAMS, EVAL_TF, DEVICE
from src.xai.per_stream_gradcam import PerStreamExplainer

IMG_DIR = Path(r'C:\Users\harry\OneDrive\Desktop\CHAT-GPT_IMAGE_2.0')
OUT_DIR = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L_chatgpt_image2')
HEATMAP_DIR = OUT_DIR / 'heatmaps'
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)


# 漏判 4 張 + 成功對照 3 張
FAILED_INDICES  = [15, 21, 23, 25]  # 1-based 索引（按 sorted 排序）
SUCCESS_INDICES = [1, 6, 12]


def main():
    images = sorted([p for p in IMG_DIR.iterdir()
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')])
    print(f"[load] Found {len(images)} images")

    # 載入主模型 + Explainer
    print("[load] Loading AIDetector...")
    t0 = time.time()
    det = AIDetector()
    print(f"[load] Done in {time.time()-t0:.1f}s")
    explainer = det.explainer

    selected = sorted(set(FAILED_INDICES + SUCCESS_INDICES))

    results = []
    for idx in selected:
        if idx > len(images):
            continue
        img_path = images[idx - 1]
        category = 'failed' if idx in FAILED_INDICES else 'success'
        print(f"\n[{idx:2d}] {category.upper():7s}  {img_path.name[:50]}")

        # 讀圖
        img_pil = Image.open(img_path).convert('RGB')
        img_224 = img_pil.resize((224, 224))
        img_np = np.array(img_224)
        img_tensor = EVAL_TF(img_pil).unsqueeze(0).to(DEVICE)

        # 1) 跑主模型推論（取 logits, fake_prob, energy）
        with torch.no_grad():
            feats = []
            for s in STREAMS:
                f_ = det.extractors[s].extract_features(img_tensor)
                feats.append(f_)
            fused = torch.cat(feats, dim=1)
            lb, _, _, _ = det.fusion(fused, grl_lambda=0)
            prob_fake_T1 = F.softmax(lb / 3.0, dim=1)[0, 1].item()  # 與 detector.py 同設
            fake_prob_T1 = F.softmax(lb, dim=1)[0, 1].item()         # T=1
            # Energy score
            energy = -torch.logsumexp(lb / 1.0, dim=1)[0].item()    # 越高越 OOD
            energy_sig = 1 / (1 + np.exp(-energy))                  # 歸一化

        # 2) 跑 Per-Stream Grad-CAM
        explanations = explainer.explain_image(img_tensor)

        # 3) 視覺化儲存
        save_path = HEATMAP_DIR / f'{category}_{idx:02d}_{img_path.stem[-12:]}.png'
        explainer.visualize(
            img_np, explanations,
            save_path=str(save_path),
            title=f'EXP-L Image #{idx} ({category})  | fusion fake_prob (T=3) = {prob_fake_T1*100:.2f}%  | Energy = {energy:.3f}'
        )

        results.append({
            'idx': idx,
            'category': category,
            'filename': img_path.name,
            'fake_prob_T1':  round(fake_prob_T1 * 100, 2),
            'fake_prob_T3':  round(prob_fake_T1 * 100, 2),
            'energy_score':  round(energy, 4),
            'energy_sig':    round(energy_sig, 4),
            'heatmap_path':  str(save_path),
        })
        print(f"      fake_prob(T=1)={fake_prob_T1*100:6.2f}% | fake_prob(T=3)={prob_fake_T1*100:6.2f}% | Energy={energy:7.3f}")

    # 4) Energy 補救分析：用 fake_prob_T1 + α·sigmoid(energy)
    print()
    print("=" * 80)
    print("  Energy 補救分析：S_final = S_main(T=1) + α · sigmoid(energy)")
    print("=" * 80)
    print(f"  {'idx':>3s} {'cat':>7s} {'S_main':>8s} {'+α=0.05':>9s} {'+α=0.10':>9s} {'+α=0.20':>9s} {'救回?':>6s}")
    for r in results:
        if r['category'] != 'failed':
            continue
        s_main = r['fake_prob_T1'] / 100
        for alpha in (0.05, 0.10, 0.20):
            pass  # 下面列出
        s05 = s_main + 0.05 * r['energy_sig']
        s10 = s_main + 0.10 * r['energy_sig']
        s20 = s_main + 0.20 * r['energy_sig']
        rescued = '是' if s10 > 0.5 else ('微' if s20 > 0.5 else '否')
        print(f"  #{r['idx']:>2d}  failed  {s_main*100:7.2f}%  {s05*100:8.2f}%  {s10*100:8.2f}%  {s20*100:8.2f}%   {rescued}")

    # 5) 儲存完整結果
    out_json = OUT_DIR / 'energy_analysis.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_json}")


if __name__ == '__main__':
    main()
