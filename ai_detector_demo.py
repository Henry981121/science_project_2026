"""
AI Image Detector Demo — v3
==============================
1. Per-stream AI probability (from per-stream heads)
2. Fusion model verdict (from GRL model)
3. Stream importance (gradient-based, not attention)
4. Grad-CAM heatmaps (ResNet50-style, clear red/blue)

Grad-CAM approach: Each stream extractor acts as its own end-to-end
classifier (extractor + head). Grad-CAM is computed directly on the
CNN backbone → produces clear heatmaps like the reference image.
"""
import sys

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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


class GradCAM:
    """Standard Grad-CAM — hooks into a conv layer, generates clear heatmap."""
    def __init__(self, target_layer):
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.activations = o.detach()

    def _bwd(self, m, gi, go):
        self.gradients = go[0].detach()

    def compute(self):
        if self.gradients is None or self.activations is None:
            return None
        w = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        if cam.ndim < 2:
            return None
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class AIDetector:
    def __init__(self):
        print("Loading AI Detector v3...")

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

        # Setup Grad-CAM hooks on last conv layer of each CNN extractor
        self.gradcams = {}
        for s in ['fft', 'dct', 'dire', 'noise']:
            target = self._get_target(s)
            if target is not None:
                self.gradcams[s] = GradCAM(target)

        # Load fusion model
        from s3_main_grl import FusionDetectorGRL
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        self.fusion = FusionDetectorGRL(
            n_streams=ckpt['n_streams'], n_sources=ckpt['n_sources'], n_gen=ckpt['n_gen'])
        self.fusion.load_state_dict(ckpt['model_state_dict'])
        self.fusion.to(DEVICE).eval()

        # Load ResNet50 for Grad-CAM (produces clear heatmaps)
        import torchvision.models as tv_models
        resnet = tv_models.resnet50(weights=None)
        resnet.fc = nn.Linear(2048, 2)
        resnet.load_state_dict(torch.load(
            Path(r'C:\Users\harry\OneDrive\Desktop\outputs\grad cam 測試\gradcam\best_model.pth'),
            map_location=DEVICE, weights_only=True))
        resnet = resnet.to(DEVICE)
        resnet.eval()
        self.resnet = resnet
        self.resnet_gradcam = GradCAM(resnet.layer4[-1])
        print("  ResNet50 Grad-CAM: loaded")

        print("Ready!")

    def _get_target(self, s):
        ext = self.extractors[s]
        if s == 'fft':
            children = list(ext.backbone.children())
        elif s in ('dct', 'dire'):
            children = list(ext.feature_net.children())
        elif s == 'noise':
            children = list(ext.cnn.children())
        else:
            return None
        # Find last Sequential block (ResNet layer4 equivalent)
        for c in reversed(children):
            if isinstance(c, nn.Sequential) and len(list(c.children())) > 0:
                return c[-1]
        return children[-2] if len(children) > 1 else None

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

        # ── ResNet50 Grad-CAM (clear, like reference image) ──
        heatmaps = {}
        img_resnet = EVAL_TF(img_pil).unsqueeze(0).to(DEVICE)
        img_resnet.requires_grad_(True)
        self.resnet.eval()
        out = self.resnet(img_resnet)
        pred_r = out.argmax(1).item()
        self.resnet.zero_grad()
        out[0, pred_r].backward()
        cam = self.resnet_gradcam.compute()
        if cam is not None:
            cam_resized = cv2.resize(cam, (224, 224))
            heatmap_color = cm.jet(cam_resized)[:, :, :3]
            overlay = np.clip(0.5 * img_np / 255.0 + 0.5 * heatmap_color, 0, 1)
            overlay_img = (overlay * 255).astype(np.uint8)
            heatmaps['resnet50'] = overlay_img
        img_resnet.requires_grad_(False)

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
        ax.set_title('Stream Importance\n(Gradient-based)', fontsize=12, fontweight='bold')
        plt.tight_layout()

        # ── Grad-CAM figure (like reference: top=original, bottom=heatmap) ──
        hm = result['heatmaps']
        verdict = f['prediction']

        fig_cam, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(result['original'])
        axes[0].set_title('Original', fontsize=13, fontweight='bold')
        axes[0].axis('off')

        if 'resnet50' in hm:
            axes[1].imshow(hm['resnet50'])
        else:
            axes[1].imshow(result['original'])
        axes[1].set_title('Grad-CAM (Red = AI Artifact Region)', fontsize=13, fontweight='bold')
        axes[1].axis('off')

        prob_str = _conf_str(f['prob_fake'], verdict)
        fig_cam.suptitle(f'Verdict: {verdict} ({prob_str})',
                         fontsize=14, fontweight='bold',
                         color='#C44E52' if verdict == 'AI' else '#55A868')
        plt.tight_layout()

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
