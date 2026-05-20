"""
AI Image Detector Demo — v4
==============================
1. Per-stream AI probability (from per-stream heads)
2. Fusion model verdict (from GRL model)
3. Stream importance (from cross-attention weights)
4. Per-stream Grad-CAM heatmaps (真正對應五流模型，非獨立 ResNet50)

Grad-CAM 由 src/xai/per_stream_gradcam.py 的 PerStreamExplainer 提供：
  - FFT / DCT / DIRE / Noise: 真 Grad-CAM (Selvaraju 2017)，target 為各
    backbone 最後 conv block，backward target 為該流 EXP-A head 的 fake logit
  - CLIP: 暫用 attention rollout（Day 2-3 會升級成 Chefer relevance）
"""
import sys

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.xai.per_stream_gradcam import PerStreamExplainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from config import OUTPUTS_DIR, FEAT_CACHE_DIR
MODEL_PATH = OUTPUTS_DIR / 'main_grl' / 'best_model.pth'
EXP_A_DIR = OUTPUTS_DIR / 'exp_a'
FEAT_DIR = FEAT_CACHE_DIR

STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']
STREAM_DISPLAY = {'clip': 'CLIP', 'fft': 'FFT', 'dct': 'DCT', 'dire': 'DIRE', 'noise': 'Noise'}
TEMPERATURE = 3.0

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    def forward(self, x):
        return self.net(x)


class AIDetector:
    def __init__(self):
        print("Loading AI Detector v4...")

        from src.feature_extractors import (
            CLIPFeatureExtractor, FFTFeatureExtractor,
            DCTFeatureExtractor, DIREFeatureExtractor, NoisePrintExtractor,
        )

        ext_classes = {
            'clip': CLIPFeatureExtractor, 'fft': FFTFeatureExtractor,
            'dct': DCTFeatureExtractor, 'dire': DIREFeatureExtractor,
            'noise': NoisePrintExtractor,
        }

        # Load extractors with saved weights
        self.extractors = {}
        for s, cls in ext_classes.items():
            ext = cls(device='cpu')
            wp = FEAT_DIR / f"{s}_extractor.pth"
            if wp.exists():
                ext.load_state_dict(torch.load(wp, map_location='cpu', weights_only=False), strict=False)
            ext.to(DEVICE)
            # Need gradients for Grad-CAM
            for p in ext.parameters():
                p.requires_grad_(True)
            self.extractors[s] = ext
            print(f"  {s}: loaded")

        # Load per-stream heads
        self.heads = {}
        for s in STREAMS:
            hp = EXP_A_DIR / s / 'best_model.pth'
            if hp.exists():
                h = LinearHead().to(DEVICE)
                h.load_state_dict(torch.load(hp, weights_only=False))
                for p in h.parameters():
                    p.requires_grad_(True)
                self.heads[s] = h

        # Load fusion model
        from s3_main_grl import FusionDetectorGRL
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        self.fusion = FusionDetectorGRL(
            n_streams=ckpt['n_streams'], n_sources=ckpt['n_sources'], n_gen=ckpt['n_gen'])
        self.fusion.load_state_dict(ckpt['model_state_dict'])
        self.fusion.to(DEVICE).eval()

        # Per-stream Grad-CAM explainer — 真正對應五流模型
        self.explainer = PerStreamExplainer(self.extractors, self.heads, device=DEVICE)
        print("  PerStreamExplainer: ready")

        print("Ready!")

    def analyze_image(self, image_pil):
        img_pil = image_pil.convert('RGB')
        img_224 = img_pil.resize((224, 224))
        img_np = np.array(img_224)
        img_tensor = EVAL_TF(img_pil)
        img_batch = img_tensor.unsqueeze(0).to(DEVICE)

        # ── Per-stream predictions ──
        stream_results = {}
        stream_importance = {}

        for s in STREAMS:
            if s not in self.heads:
                continue
            with torch.no_grad():
                feat = self.extractors[s].extract_features(img_batch)
                logits = self.heads[s](feat)
                prob = F.softmax(logits / TEMPERATURE, dim=1)[0, 1].item()
                pred = logits.argmax(1).item()
            stream_results[s] = {
                'prob_fake': round(prob * 100, 2),
                'prediction': 'AI' if pred == 1 else 'Real',
            }
            # Use raw logit magnitude as importance proxy
            stream_importance[s] = abs(logits[0, 1].item() - logits[0, 0].item())

        # ── Normalize stream importance ──
        total_imp = sum(stream_importance.values()) + 1e-8
        stream_weights = {s: round(v / total_imp * 100, 1) for s, v in stream_importance.items()}

        # ── Per-stream Grad-CAM (FFT/DCT/DIRE/Noise + CLIP) ──
        # 對應到融合模型每條流，非獨立 ResNet50
        heatmaps = self.explainer.explain_image(img_batch)

        # ── Fusion prediction + Attention Weights ──
        with torch.no_grad():
              feats = []
              for s in STREAMS:
                  feats.append(self.extractors[s].extract_features(img_batch))
              fused = torch.cat(feats, dim=1)
              lb, _, _, attn_weights = self.fusion(fused, grl_lambda=0)
              fusion_prob = F.softmax(lb / TEMPERATURE, dim=1)[0, 1].item()
              fusion_pred = lb.argmax(1).item()

              # CLIP override: when CLIP is very confident (>95%) but Fusion disagrees,
              # blend CLIP's opinion into the final result.
              # This prevents DCT/Noise false positives on phone photos.
              clip_prob = stream_results.get('clip', {}).get('prob_fake', 50) / 100
              clip_conf = max(clip_prob, 1 - clip_prob)  # confidence regardless of direction
              if clip_conf > 0.90:  # CLIP is very sure
                  # Weighted blend: 60% Fusion + 40% CLIP
                  blended_prob = 0.6 * fusion_prob + 0.4 * clip_prob
                  fusion_prob = blended_prob
                  fusion_pred = 1 if fusion_prob > 0.5 else 0

              # 用 Cross-Attention weights 當 stream 重要性
              if attn_weights is not None:
                  w = attn_weights[0].mean(dim=0).cpu().numpy()
                  w = w / (w.sum() + 1e-8) * 100
                  stream_weights = {s: round(float(w[i]), 1) for i, s in enumerate(STREAMS) if i < len(w)}
              else:
                  stream_weights = {s: 20.0 for s in STREAMS}

        return {
              'streams': stream_results,
              'fusion': {
                  'prob_fake': round(fusion_prob * 100, 2),
                  'prediction': 'AI' if fusion_pred == 1 else 'Real',
              },
              'stream_weights': stream_weights,
              'heatmaps': heatmaps,
              'original': img_np,
          }
        


def _conf_str(prob_fake, prediction):
    if prediction == 'AI':
        return f"{prob_fake:.1f}% AI"
    return f"{100-prob_fake:.1f}% Real"


def create_gradio_app(detector):
    import gradio as gr

    def predict(image):
        if image is None:
            return "Upload an image.", None, None

        result = detector.analyze_image(image)
        f = result['fusion']

        # ── Text ──
        lines = ["=" * 50, "  AI Image Detection Result", "=" * 50]
        lines.append(f"  VERDICT: {f['prediction']}  ({_conf_str(f['prob_fake'], f['prediction'])})")
        lines.append("")
        lines.append(f"  {'Stream':<8} {'Confidence':>12} {'Weight':>8} {'Verdict':>8}")
        lines.append(f"  {'-'*40}")
        for s in STREAMS:
            if s in result['streams']:
                r = result['streams'][s]
                w = result['stream_weights'].get(s, 0)
                lines.append(f"  {STREAM_DISPLAY[s]:<8} {_conf_str(r['prob_fake'], r['prediction']):>12} {w:>6.1f}%  {r['prediction']:>6}")
        lines.append(f"  {'─'*40}")
        lines.append(f"  {'Fusion':<8} {_conf_str(f['prob_fake'], f['prediction']):>12}          {f['prediction']:>6}")
        lines.append("=" * 50)
        text = '\n'.join(lines)

        # ── Stream weight pie ──
        sw = result['stream_weights']
        fig_w, ax = plt.subplots(figsize=(5, 5))
        labels = [STREAM_DISPLAY[s] for s in STREAMS if s in sw]
        sizes = [sw[s] for s in STREAMS if s in sw]
        colors_p = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_p[:len(labels)],
               startangle=90, textprops={'fontsize': 11})
        ax.set_title('Stream Importance\n(Cross-Attention)', fontsize=12, fontweight='bold')
        plt.tight_layout()

        # ── Per-stream Grad-CAM figure: 原圖 + 5 條流 ──
        verdict = f['prediction']
        prob_str = _conf_str(f['prob_fake'], verdict)
        fig_cam = detector.explainer.visualize(
            result['original'],
            result['heatmaps'],
            title=f'Verdict: {verdict} ({prob_str})',
        )

        return text, fig_w, fig_cam

    with gr.Blocks(title="AI Image Detector") as demo:
        gr.Markdown("# AI Image Detector (Multi-Stream + GRL Fusion)")
        gr.Markdown(
            "Upload an image to detect if it is AI-generated.\n\n"
            "**Best for:** Images downloaded from the internet, social media, AI art platforms (Midjourney, Stable Diffusion, DALL-E)\n\n"
            "**Note:** Mobile phone photos, screenshots, or heavily cropped/rotated images may produce less accurate results."
        )

        img_input = gr.Image(type="pil", label="Upload Image")
        btn = gr.Button("Analyze", variant="primary")

        txt_out = gr.Textbox(label="Detection Result", lines=14)
        with gr.Row():
            plot_w = gr.Plot(label="Stream Importance")
        plot_cam = gr.Plot(label="Grad-CAM Heatmaps")

        btn.click(predict, inputs=img_input, outputs=[txt_out, plot_w, plot_cam])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?')
    parser.add_argument('--web', action='store_true')
    args = parser.parse_args()

    detector = AIDetector()

    if args.web or not args.image:
        app = create_gradio_app(detector)
        app.launch(share=False, server_name="127.0.0.1", server_port=7860)
    else:
        result = detector.analyze_image(Image.open(args.image))
        f = result['fusion']
        print(f"\nVerdict: {f['prediction']} ({_conf_str(f['prob_fake'], f['prediction'])})")
        for s in STREAMS:
            if s in result['streams']:
                r = result['streams'][s]
                w = result['stream_weights'].get(s, 0)
                print(f"  {STREAM_DISPLAY[s]:<8} {_conf_str(r['prob_fake'], r['prediction']):>12}  weight={w:.1f}%")


if __name__ == '__main__':
    main()
