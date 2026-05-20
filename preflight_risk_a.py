"""
preflight_risk_a.py
====================

【目的】
測試五條 stream extractor 的「梯度路徑」有沒有斷掉。
如果某條流的 extract_features() 內部有 @torch.no_grad() 或 .detach()，
梯度就反不回 backbone 的 conv 層 → Grad-CAM 失效。

這支 script 用黑箱測試方式驗證：
  1. 餵一張 requires_grad=True 的假圖進 extractor
  2. 檢查輸出 feat 有沒有 grad_fn（沒斷的話會有）
  3. 對 backbone 的 target conv 層掛 hook，跑 backward
  4. 檢查 target 層有沒有收到 gradient

【怎麼用】
從專案 root 跑：
    python preflight_risk_a.py

【結果判讀】
- 全 PASS  → Day 1 可直接動工，不用改 extractor
- 有 FAIL  → 對應 extractor 內部有梯度阻擋，需要：
             (a) 找出 @torch.no_grad() decorator 拿掉，或
             (b) 加一個 xai_mode=True 的參數讓 extract_features 跳過 no_grad，
             不要直接改訓練流程那條路（會影響 inference 效能）

【作者】XAI 重做計畫 - Pre-flight check A
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from src.feature_extractors import (
    CLIPFeatureExtractor, FFTFeatureExtractor,
    DCTFeatureExtractor, DIREFeatureExtractor, NoisePrintExtractor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXTRACTOR_CLASSES = {
    'clip':  CLIPFeatureExtractor,
    'fft':   FFTFeatureExtractor,
    'dct':   DCTFeatureExtractor,
    'dire':  DIREFeatureExtractor,
    'noise': NoisePrintExtractor,
}


def get_target_module(ext, name):
    """
    找該流的 Grad-CAM target layer（backbone 最後一個 Sequential block）。
    這段邏輯直接從 ai_detector_demo.py:150-164 的 _get_target() 搬過來。
    CLIP 是 ViT，沒有 conv target，直接 return None。
    """
    if name == 'clip':
        return None
    if name == 'fft':
        children = list(ext.backbone.children())
    elif name in ('dct', 'dire'):
        children = list(ext.feature_net.children())
    elif name == 'noise':
        children = list(ext.cnn.children())
    else:
        return None
    for c in reversed(children):
        if isinstance(c, nn.Sequential) and len(list(c.children())) > 0:
            return c[-1]
    return children[-2] if len(children) > 1 else None


def test_extractor(name, cls):
    print(f"\n=== {name.upper()} ===")
    try:
        ext = cls(device='cpu').to(DEVICE)
    except Exception as e:
        print(f"  FAIL to instantiate: {e}")
        return False

    # 開啟 extractor 內所有參數的 grad
    for p in ext.parameters():
        p.requires_grad_(True)

    # ─── Test 1: extract_features() 是否保留 grad_fn ───
    img = torch.randn(1, 3, 224, 224, device=DEVICE, requires_grad=True)
    try:
        feat = ext.extract_features(img)
    except Exception as e:
        print(f"  FAIL extract_features: {e}")
        return False

    has_grad_fn = feat.grad_fn is not None
    print(f"  feat.shape         = {tuple(feat.shape)}")
    print(f"  feat.requires_grad = {feat.requires_grad}")
    print(f"  feat.grad_fn       = {feat.grad_fn}")

    if not has_grad_fn:
        # grad_fn 是 None → extract_features 內部把梯度斷掉了
        print(f"  >>> FAIL: gradient broken inside extract_features()")
        print(f"           檢查該 extractor 是否有 @torch.no_grad() 或 .detach()")
        return False

    # ─── Test 2: backward 後 target conv 層有沒有收到 gradient ───
    target = get_target_module(ext, name)
    if target is None:
        # CLIP 走 attention rollout / Chefer，不需要 conv target
        print(f"  >>> PASS (ViT, no conv target needed)")
        return True

    activations, gradients = {}, {}
    def fwd_hook(m, i, o):
        activations['v'] = o
    def bwd_hook(m, gi, go):
        gradients['v'] = go[0]

    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)

    try:
        # 重新 forward 一次（前一次沒掛 hook）
        img2 = torch.randn(1, 3, 224, 224, device=DEVICE, requires_grad=True)
        feat2 = ext.extract_features(img2)
        feat2.sum().backward()
    except Exception as e:
        print(f"  FAIL during backward: {e}")
        h1.remove(); h2.remove()
        return False

    h1.remove(); h2.remove()

    if 'v' not in activations:
        # _get_target 挑錯模組，target 從未被呼叫
        print(f"  >>> FAIL: target module never fired")
        print(f"           (_get_target 選錯模組，要修 get_target_module())")
        return False
    if 'v' not in gradients or gradients['v'] is None:
        print(f"  >>> FAIL: no gradient reached target conv layer")
        return False

    print(f"  target             = {target.__class__.__name__}")
    print(f"  activation.shape   = {tuple(activations['v'].shape)}")
    print(f"  gradient.shape     = {tuple(gradients['v'].shape)}")
    print(f"  >>> PASS")
    return True


if __name__ == "__main__":
    print("="*60)
    print("  Pre-flight Risk A: Gradient flow check for Grad-CAM")
    print("="*60)
    print(f"  Device: {DEVICE}")

    results = {n: test_extractor(n, c) for n, c in EXTRACTOR_CLASSES.items()}

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for n, ok in results.items():
        print(f"  {n:8s} : {'PASS' if ok else 'FAIL'}")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\n  Failed streams : {failed}")
        print(f"  Action         : 開啟對應的 extractor 檔案，")
        print(f"                   拿掉 extract_features() 路徑上的")
        print(f"                   @torch.no_grad() decorator 或 .detach() 呼叫。")
        print(f"                   建議加 xai_mode=True 參數，避免影響訓練/推論效能。")
        sys.exit(1)
    else:
        print(f"\n  All streams PASS. Day 1 implementation can proceed unmodified.")
        sys.exit(0)
