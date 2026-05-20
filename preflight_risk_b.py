"""
preflight_risk_b.py
====================

【目的】
檢查 fusion 模型的 cross-attention 分布**有沒有隨 generator 類型變化**。
這決定計畫裡 Level 2 的敘事走向：
  - 如果「GAN 圖 → FFT/DCT 權重高、Diffusion 圖 → DIRE 權重高」這個模式真的存在
    → Level 2 可以直接寫「我們驗證了多流互補」
  - 如果模式不存在
    → Level 2 敘事要改成中性描述（避免被打臉）

【做什麼】
1. 載入訓好的 fusion 模型
2. 拿快取的 test features 跑一次 inference，收集每張圖的 attention 權重
3. 按 generator 類型分組求平均
4. 印出三種視角：
     a) 按家族（Real / GAN / Diffusion / Other）
     b) 按具體 generator（樣本數 >= 50 的才列）
     c) 自動判斷「預期模式」是否成立
5. 把每張圖的 attention 存成 CSV，供 Phase 1 的 Level 2 圖直接用

【怎麼用】
從專案 root 跑：
    python preflight_risk_b.py

【結果判讀】
- 印出 `PATTERN HOLDS` → Level 2 敘事照計畫寫
- 印出 `PATTERN BROKEN` → 看上面的家族表，自己看實際模式是什麼，回頭改計畫

【可能會卡住的地方】
1. config.py 沒有 TEST_CSV 變數
     → 在下面 "LOCATE TEST CSV" 區手動填路徑
2. 測試集 CSV 的 generator 欄位名稱不是 'generator'
     → 改下面的 df['generator'] 為實際欄位名
3. Generator 名字不在 family() 的關鍵字裡（例如 firefly、ideogram）
     → 在 family() 函數內加關鍵字
4. Fusion 的 attn shape 不是 (B, heads, N, N) 或 (B, N, N)
     → 看 print 出來的 shape，調整下面的 aggregation 邏輯

【輸出檔案】
    OUTPUTS_DIR/preflight_attn.csv
    (Phase 1 Level 2 畫圖直接讀這個 csv，不用重跑 inference)

【作者】XAI 重做計畫 - Pre-flight check B
"""

import torch
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from config import OUTPUTS_DIR, FEAT_CACHE_DIR
from s3_main_grl import FusionDetectorGRL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = OUTPUTS_DIR / 'main_grl' / 'best_model.pth'
STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']


# ══════════════════════════════════════════════════════════════
# 1. 載入 fusion 模型
# ══════════════════════════════════════════════════════════════
print("="*60)
print("  Pre-flight Risk B: Fusion attention pattern check")
print("="*60)
print(f"  Device     : {DEVICE}")
print(f"  Model path : {MODEL_PATH}")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = FusionDetectorGRL(
    n_streams=ckpt['n_streams'],
    n_sources=ckpt['n_sources'],
    n_gen=ckpt['n_gen'],
)
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE).eval()
print(f"  Model loaded.")


# ══════════════════════════════════════════════════════════════
# 2. 載入快取的 test features
# ══════════════════════════════════════════════════════════════
print("\nLoading cached test features...")
test_feats = torch.cat([
    torch.load(FEAT_CACHE_DIR / f"{s}_test_feats.pt", weights_only=False)
    for s in STREAMS
], dim=1)
print(f"  test_feats.shape = {tuple(test_feats.shape)}")


# ══════════════════════════════════════════════════════════════
# 3. 找 test CSV
# ══════════════════════════════════════════════════════════════
import config
TEST_CSV = getattr(config, 'TEST_CSV', None)

# ↓↓↓ 如果 config.py 裡沒有 TEST_CSV，把實際路徑填在這裡 ↓↓↓
# TEST_CSV = '/path/to/test.csv'
# ↑↑↑ ──────────────────────────────────────────────────── ↑↑↑

if TEST_CSV is None:
    raise RuntimeError(
        "TEST_CSV 未定義。請在 config.py 加一行 TEST_CSV = '...'，"
        "或直接在這支 script 上方手動填路徑。"
    )

df = pd.read_csv(TEST_CSV)
print(f"\nTest CSV : {TEST_CSV}")
print(f"  rows   : {len(df)}")
print(f"  cols   : {list(df.columns)}")

# 對齊長度（如果不一致表示 cache 跟 csv 不同步）
if len(df) != len(test_feats):
    raise RuntimeError(
        f"CSV ({len(df)}) 跟快取 features ({len(test_feats)}) 長度不一致。"
        "可能是 cache 過期，需要重跑 feature 提取。"
    )

# generator 欄位名 — 如果你的 csv 不是這個欄位名，改下面這行
GEN_COL = 'generator'
if GEN_COL not in df.columns:
    raise RuntimeError(
        f"CSV 找不到 '{GEN_COL}' 欄位。可用欄位：{list(df.columns)}。"
        "請改本 script 的 GEN_COL 變數。"
    )

print(f"\nGenerator 分布：")
print(df[GEN_COL].value_counts())


# ══════════════════════════════════════════════════════════════
# 4. 收集每張圖的 attention 權重
# ══════════════════════════════════════════════════════════════
print("\nCollecting attention weights...")
all_w = []
with torch.no_grad():
    for i in range(0, len(test_feats), 256):
        batch = test_feats[i:i+256].to(DEVICE)
        _, _, _, attn = model(batch, grl_lambda=0)
        if attn is None:
            raise RuntimeError("Fusion model 沒有回傳 attention weights")

        # attn 可能是 (B, heads, N, N) 或 (B, N, N)
        # 目標：壓成 (B, N) — 每張圖對每條流的 attention 分布
        # 邏輯參考 ai_detector_demo.py:236-239
        a = attn.cpu().numpy()
        if a.ndim == 4:
            a = a.mean(axis=1)    # (B, heads, N, N) → (B, N, N)，平均 head
        w = a.mean(axis=1)        # (B, N, N) → (B, N)，平均 query token
        all_w.append(w)

W = np.concatenate(all_w, axis=0)
W = W / (W.sum(axis=1, keepdims=True) + 1e-8)  # normalize 成機率分布
print(f"Attention matrix shape: {W.shape}")

# 寫回 dataframe
for i, s in enumerate(STREAMS):
    df[f'attn_{s}'] = W[:, i]


# ══════════════════════════════════════════════════════════════
# 5. 按家族分組（Real / GAN / Diffusion / Other）
# ══════════════════════════════════════════════════════════════
def family(g):
    """
    根據 generator 名稱分到家族。
    如果你的 dataset 有新 generator 名字（例如 firefly、ideogram），
    在這裡加關鍵字。
    """
    g = str(g).lower()
    if g in ('real', 'natural', 'none', 'nan', ''):
        return 'Real'
    if any(k in g for k in ['stylegan', 'progan', 'biggan', 'gan']):
        return 'GAN'
    if any(k in g for k in ['stable', 'sd', 'diffusion', 'dalle', 'midjourney',
                            'mj', 'glide', 'imagen', 'sdxl', 'firefly']):
        return 'Diffusion'
    return 'Other'

df['family'] = df[GEN_COL].apply(family)

print("\n" + "="*60)
print("  Per-FAMILY mean attention (%)")
print("="*60)
fam = df.groupby('family')[[f'attn_{s}' for s in STREAMS]].mean()
print((fam * 100).round(1))


# ══════════════════════════════════════════════════════════════
# 6. 按具體 generator 分組（n >= 50 才列出來）
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Per-GENERATOR mean attention (% — n >= 50 only)")
print("="*60)
counts = df[GEN_COL].value_counts()
big = counts[counts >= 50].index
gen_table = df[df[GEN_COL].isin(big)].groupby(GEN_COL)[
    [f'attn_{s}' for s in STREAMS]
].mean()
print((gen_table * 100).round(1))


# ══════════════════════════════════════════════════════════════
# 7. 判斷「預期模式」是否成立
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  VERDICT on Level 2 narrative")
print("="*60)
fam_pct = fam * 100

if 'GAN' in fam_pct.index and 'Diffusion' in fam_pct.index:
    gan = fam_pct.loc['GAN']
    dif = fam_pct.loc['Diffusion']

    # 預期：GAN 圖 → FFT+DCT 高於 DIRE
    freq_dom_in_gan = (gan['attn_fft'] + gan['attn_dct']) > gan['attn_dire']
    # 預期：Diffusion 圖 → DIRE 高於 FFT 和 DCT
    dire_dom_in_dif = dif['attn_dire'] > max(dif['attn_fft'], dif['attn_dct'])

    print(f"  GAN       : FFT+DCT = {gan['attn_fft']+gan['attn_dct']:5.1f}%   "
          f"DIRE = {gan['attn_dire']:5.1f}%")
    print(f"  Diffusion : DIRE    = {dif['attn_dire']:5.1f}%   "
          f"FFT+DCT = {dif['attn_fft']+dif['attn_dct']:5.1f}%")

    if freq_dom_in_gan and dire_dom_in_dif:
        print("\n  >>> PATTERN HOLDS")
        print("  >>> Level 2 narrative is supported — 照計畫寫")
    else:
        print("\n  >>> PATTERN BROKEN")
        print("  >>> 預期模式不成立，回頭看家族表，根據實際模式改 Level 2 敘事")
        print("  >>> 不一定是壞事 — 可能模型學到的是不同的互補方式")
else:
    print("  測試集裡 GAN 或 Diffusion 樣本數不足，無法評估")


# ══════════════════════════════════════════════════════════════
# 8. 存檔給 Phase 1 用
# ══════════════════════════════════════════════════════════════
out_path = OUTPUTS_DIR / 'preflight_attn.csv'
df.to_csv(out_path, index=False)
print(f"\nSaved per-image attention to: {out_path}")
print("(Phase 1 Level 2 畫圖時直接讀這個 CSV，不用再跑一次 inference)")
print("\nDONE")
