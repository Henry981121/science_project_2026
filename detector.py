"""
AI Image Detector — 核心模組
================================
從 tech_project/ai_detector_demo.py 整合而來，移除 Windows 路徑依賴。

使用前請設定下方 ── 路徑設定 ── 區塊中的路徑，指向你的模型權重檔案。
"""

import sys
import io
import base64
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.cm as cm

# ── 路徑設定 ────────────────────────────────────────────────────────
TECH_PROJECT_DIR = Path(r'C:\Users\harry\OneDrive\Desktop\新code 3.17')
OUTPUTS_DIR      = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\3.22output')

MODEL_PATH       = OUTPUTS_DIR / 'main_grl' / 'best_model.pth'
EXP_A_DIR        = OUTPUTS_DIR / 'exp_a'
FEAT_DIR         = OUTPUTS_DIR / 'exp_a' / 'features'
GRADCAM_PATH     = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_g_v2\checkpoints\resnet50_finetuned.pth')
# ─────────────────────────────────────────────────────────────────────

STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']
STREAM_DISPLAY = {'clip': 'CLIP', 'fft': 'FFT', 'dct': 'DCT', 'dire': 'DIRE', 'noise': 'Noise'}
TEMPERATURE = 3.0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)


class GradCAM:
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
        print('Loading AI Detector...')

        # 把 Desktop 和 tech_project 加入 import 路徑
        sys.path.insert(0, r'C:\Users\harry\OneDrive\Desktop')
        sys.path.insert(0, str(TECH_PROJECT_DIR))

        from src.feature_extractors import (
            CLIPFeatureExtractor, FFTFeatureExtractor,
            DCTFeatureExtractor, DIREFeatureExtractor, NoisePrintExtractor,
        )

        ext_classes = {
            'clip': CLIPFeatureExtractor, 'fft': FFTFeatureExtractor,
            'dct': DCTFeatureExtractor,   'dire': DIREFeatureExtractor,
            'noise': NoisePrintExtractor,
        }

        # 載入特徵提取器
        self.extractors = {}
        for s, cls in ext_classes.items():
            ext = cls(device='cpu')
            wp = FEAT_DIR / f'{s}_extractor.pth'
            if wp.exists():
                ext.load_state_dict(
                    torch.load(wp, map_location='cpu', weights_only=False), strict=False
                )
            ext.to(DEVICE)
            for p in ext.parameters():
                p.requires_grad_(True)
            self.extractors[s] = ext
            print(f'  {s}: loaded')

        # 載入 per-stream heads
        self.heads = {}
        for s in STREAMS:
            hp = EXP_A_DIR / s / 'best_model.pth'
            if hp.exists():
                h = LinearHead().to(DEVICE)
                h.load_state_dict(torch.load(hp, weights_only=False))
                for p in h.parameters():
                    p.requires_grad_(True)
                self.heads[s] = h

        # 設定 Grad-CAM hooks
        self.gradcams = {}
        for s in ['fft', 'dct', 'dire', 'noise']:
            target = self._get_target(s)
            if target is not None:
                self.gradcams[s] = GradCAM(target)

        # 載入 Fusion 模型
        from s3_main_grl import FusionDetectorGRL
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        self.fusion = FusionDetectorGRL(
            n_streams=ckpt['n_streams'],
            n_sources=ckpt['n_sources'],
            n_gen=ckpt['n_gen'],
        )
        self.fusion.load_state_dict(ckpt['model_state_dict'])
        self.fusion.to(DEVICE).eval()

        # 載入 ResNet50（Grad-CAM 用）
        import torchvision.models as tv_models
        resnet = tv_models.resnet50(weights=None)
        resnet.fc = nn.Linear(2048, 2)
        resnet.load_state_dict(
            torch.load(GRADCAM_PATH, map_location=DEVICE, weights_only=True)
        )
        resnet.to(DEVICE).eval()
        self.resnet = resnet
        self.resnet_gradcam = GradCAM(resnet.layer4[-1])

        print('Ready!')

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
        for c in reversed(children):
            if isinstance(c, nn.Sequential) and len(list(c.children())) > 0:
                return c[-1]
        return children[-2] if len(children) > 1 else None

    def analyze_image(self, image_pil: Image.Image, enabled_streams: list = None) -> dict:
        active = [s for s in STREAMS if s in (enabled_streams or STREAMS)]
        img_pil = image_pil.convert('RGB')
        img_224 = img_pil.resize((224, 224))
        img_np = np.array(img_224)
        img_tensor = EVAL_TF(img_pil)
        img_batch = img_tensor.unsqueeze(0).to(DEVICE)

        # Per-stream 預測（只跑啟用的 streams）
        stream_results = {}
        stream_importance = {}
        for s in active:
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
            stream_importance[s] = abs(logits[0, 1].item() - logits[0, 0].item())

        total_imp = sum(stream_importance.values()) + 1e-8
        stream_weights = {s: round(v / total_imp * 100, 1) for s, v in stream_importance.items()}

        # ResNet50 Grad-CAM
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
            heatmaps['resnet50'] = (overlay * 255).astype(np.uint8)
        img_resnet.requires_grad_(False)

        # Fusion 預測（未啟用的 stream 特徵補零）
        with torch.no_grad():
            feats = []
            for s in STREAMS:
                f = self.extractors[s].extract_features(img_batch)
                feats.append(f if s in active else torch.zeros_like(f))
            fused = torch.cat(feats, dim=1)
            lb, _, _, attn_weights = self.fusion(fused, grl_lambda=0)
            fusion_prob = F.softmax(lb / TEMPERATURE, dim=1)[0, 1].item()
            fusion_pred = lb.argmax(1).item()

            # CLIP override（CLIP 信心 > 90% 時，混合 40% CLIP 意見）
            clip_prob = stream_results.get('clip', {}).get('prob_fake', 50) / 100
            if max(clip_prob, 1 - clip_prob) > 0.90:
                fusion_prob = 0.6 * fusion_prob + 0.4 * clip_prob
                fusion_pred = 1 if fusion_prob > 0.5 else 0

            # Cross-Attention weights
            if attn_weights is not None:
                w = attn_weights[0].mean(dim=0).cpu().numpy()
                w = w / (w.sum() + 1e-8) * 100
                stream_weights = {
                    s: round(float(w[i]), 1)
                    for i, s in enumerate(STREAMS) if i < len(w)
                }

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


def conf_str(prob_fake: float, prediction: str) -> str:
    if prediction == 'AI':
        return f'{prob_fake:.1f}% AI'
    return f'{100 - prob_fake:.1f}% Real'


def numpy_to_b64(img_np: np.ndarray) -> str:
    pil = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()
