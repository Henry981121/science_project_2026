"""
day1_test.py
=============

PerStreamExplainer 整合測試 — 在 demo 化之前，先用這支 script 獨立
驗證對單張圖能不能正常產出 6 格圖（原圖 + 5 條流）。

【Day 2 起 CLIP 已升級為 Chefer relevance】
這支 script 不需要任何修改，PerStreamExplainer 內部自動使用 Chefer。

【怎麼用】
從專案 root 跑：
    python day1_test.py <image_path>
    python day1_test.py <image_path> --out my_test_output.png

範例：
    python day1_test.py outputs/exp_a/some_sample.png
    python day1_test.py /path/to/any/jpg.jpg --out day1_check.png

【會做什麼】
1. 載入 5 條 extractor (跟 demo 一樣的程式碼路徑)
2. 載入 5 個 EXP-A 訓出的 head
3. 對指定圖片跑 PerStreamExplainer：
     - FFT/DCT/DIRE/Noise: Grad-CAM (Selvaraju 2017)
     - CLIP:               Chefer relevance (Chefer 2021 CVPR)
4. 存出 1×6 對照圖 (預設 day1_output.png)
5. 印出每條流的 heatmap 統計 (shape、min、max、有效流數)

【預期結果】
- 應該看到 6 格圖：原圖 + 5 條流的 heatmap
- 控制台應該印出每條流的形狀和強度
- 如果某條流 print "skip" / "failed"，看 traceback，常見原因：
    * EXP-A head 找不到 → 確認 outputs/exp_a/{stream}/best_model.pth 存在
    * CheferCLIPRelevance init failed → 檢查 transformers 版本，
      CLIPAttention 結構可能不同
    * CUDA OOM → 改用 --device cpu

【預期視覺結果（管理期待）】
根據 pre-flight B，CLIP 主導 47-57% attention。
跑出來的圖預期會是 "CLIP 那格訊號最強，其他四格相對淡"。
這不是 bug — 是模型實際分工。
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, '.')

from src.xai.per_stream_gradcam import PerStreamExplainer, STREAMS, STREAM_DISPLAY


# ──────────────────────────────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────────────────────────────
from config import OUTPUTS_DIR, FEAT_CACHE_DIR
EXP_A_DIR = OUTPUTS_DIR / 'exp_a'

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────────────
# LinearHead — 跟 ai_detector_demo.py:46-51 同結構
# ──────────────────────────────────────────────────────────────────────
class LinearHead(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────
# Load helpers
# ──────────────────────────────────────────────────────────────────────
def load_extractors(device: str):
    from src.feature_extractors import (
        CLIPFeatureExtractor, FFTFeatureExtractor,
        DCTFeatureExtractor, DIREFeatureExtractor, NoisePrintExtractor,
    )
    ext_classes = {
        'clip':  CLIPFeatureExtractor,
        'fft':   FFTFeatureExtractor,
        'dct':   DCTFeatureExtractor,
        'dire':  DIREFeatureExtractor,
        'noise': NoisePrintExtractor,
    }
    extractors = {}
    for s, cls in ext_classes.items():
        ext = cls(device='cpu')
        wp = FEAT_CACHE_DIR / f"{s}_extractor.pth"
        if wp.exists():
            ext.load_state_dict(torch.load(wp, map_location='cpu', weights_only=False),
                                strict=False)
        ext.to(device)
        for p in ext.parameters():
            p.requires_grad_(True)
        extractors[s] = ext
        print(f"  extractor [{s}]  loaded from {wp.name if wp.exists() else '(no checkpoint)'}")
    return extractors


def load_heads(device: str):
    heads = {}
    for s in STREAMS:
        hp = EXP_A_DIR / s / 'best_model.pth'
        if not hp.exists():
            print(f"  head      [{s}]  MISSING ({hp}) — skip")
            continue
        h = LinearHead().to(device)
        h.load_state_dict(torch.load(hp, weights_only=False))
        for p in h.parameters():
            p.requires_grad_(True)
        heads[s] = h
        print(f"  head      [{s}]  loaded")
    return heads


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='路徑：要解釋的測試圖片')
    parser.add_argument('--out', default='day1_output.png',
                        help='輸出 PNG 檔名（預設 day1_output.png）')
    parser.add_argument('--device', default=None,
                        help='cuda / cpu（預設自動偵測）')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("  Day 1 test — PerStreamExplainer")
    print("="*60)
    print(f"  Device : {device}")
    print(f"  Image  : {args.image}")
    print(f"  Output : {args.out}")
    print()

    # ── 1. 載入 extractor + heads ──
    print("Loading extractors...")
    extractors = load_extractors(device)
    print("\nLoading heads...")
    heads = load_heads(device)

    # ── 2. 載入圖片 ──
    img_pil = Image.open(args.image).convert('RGB')
    img_224 = img_pil.resize((224, 224))
    img_np = np.array(img_224)
    img_tensor = EVAL_TF(img_pil).unsqueeze(0).to(device)
    print(f"\nImage loaded, tensor shape = {tuple(img_tensor.shape)}")

    # ── 3. 跑 explainer ──
    print("\nRunning PerStreamExplainer...")
    explainer = PerStreamExplainer(extractors, heads, device=device)
    exps = explainer.explain_image(img_tensor)

    print(f"\nProduced heatmaps for {len(exps)}/{len(STREAMS)} streams:")
    for s in STREAMS:
        if s in exps:
            m = exps[s]
            print(f"  {STREAM_DISPLAY[s]:5s}  shape={str(tuple(m.shape)):20s}  "
                  f"min={m.min():.3f}  max={m.max():.3f}  mean={m.mean():.3f}")
        else:
            print(f"  {STREAM_DISPLAY[s]:5s}  MISSING")

    # ── 4. 視覺化 + 存檔 ──
    fig = explainer.visualize(
        img_np, exps,
        save_path=args.out,
        title=f'Day 1 Grad-CAM Test — {Path(args.image).name}',
    )

    print("\n" + "="*60)
    if len(exps) == len(STREAMS):
        print("  ALL 5 STREAMS PASS — ready for Day 2-3 (CLIP → Chefer)")
    else:
        missing = [s for s in STREAMS if s not in exps]
        print(f"  PARTIAL: missing {missing}")
        print(f"  檢查 console 上面的 warning 訊息找原因")
    print("="*60)


if __name__ == "__main__":
    main()
