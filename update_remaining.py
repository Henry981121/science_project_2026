"""Update EXP-F SOTA + t-SNE with latest 3.22 model."""
import sys, io, json, torch, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') if hasattr(sys.stdout, 'buffer') else sys.stdout
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.manifold import TSNE
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import FEAT_CACHE_DIR, OUTPUTS_DIR, SPLITS_DIR
FEAT_DIR = FEAT_CACHE_DIR
MODEL_PATH = OUTPUTS_DIR / 'main_grl' / 'best_model.pth'
OUT_DIR = OUTPUTS_DIR
STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']
device = 'cuda'

from s3_main_grl import FusionDetectorGRL
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model = FusionDetectorGRL(n_streams=ckpt['n_streams'], n_sources=ckpt['n_sources'], n_gen=ckpt['n_gen'])
model.load_state_dict(ckpt['model_state_dict'])
model.to(device).eval()

# ── EXP-F SOTA ──
print("=" * 60)
print("  EXP-F SOTA Comparison (3.22)")
print("=" * 60)

test_feats = torch.cat([torch.load(FEAT_DIR / f"{s}_test_feats.pt", weights_only=False) for s in STREAMS], dim=1)
test_labels = torch.load(FEAT_DIR / "test_labels.pt", weights_only=False)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

loader = DataLoader(TensorDataset(test_feats, test_labels), batch_size=256, shuffle=False)
all_probs, all_preds, all_labels = [], [], []
with torch.no_grad():
    for fb, lb in loader:
        logits, _, _, _ = model(fb.to(device), grl_lambda=0)
        all_probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy().tolist())
        all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
        all_labels.extend(lb.numpy().tolist())

acc = accuracy_score(all_labels, all_preds) * 100
auc = roc_auc_score(all_labels, np.array(all_probs)) * 100
f1 = f1_score(all_labels, all_preds, zero_division=0)

sota = {
    'UnivFD (ResNet50)': {'acc': 71.4, 'auc': 78.8, 'f1': 0.6243, 'params_m': 0.004},
    'EfficientNet-B4':   {'acc': 77.2, 'auc': 85.2, 'f1': 0.7087, 'params_m': 17.6},
    'Ours (GRL v2.1)':   {'acc': round(acc, 2), 'auc': round(auc, 2), 'f1': round(f1, 4), 'params_m': round(n_params, 1)},
}

with open(OUT_DIR / 'exp_f_sota_comparison.json', 'w') as f:
    json.dump(sota, f, indent=2)

for name, r in sota.items():
    print(f"  {name:<24} Acc={r['acc']:.1f}% AUC={r['auc']:.1f}% F1={r['f1']:.4f}")
print(f"  Saved!")

# ── t-SNE ──
print(f"\n{'=' * 60}")
print("  t-SNE Visualization (3.22)")
print("=" * 60)

test_df = pd.read_csv(SPLITS_DIR / 'test.csv')
cross_df = pd.read_csv(SPLITS_DIR / 'cross_generator_test.csv')
cross_feats = torch.cat([torch.load(FEAT_DIR / f"{s}_cross_gen_test_feats.pt", weights_only=False) for s in STREAMS], dim=1)
cross_labels = torch.load(FEAT_DIR / "cross_gen_test_labels.pt", weights_only=False)

n_test = min(len(test_feats), len(test_df))
n_cross = min(len(cross_feats), len(cross_df))
test_gens = test_df['generator'].astype(str).values[:n_test]
cross_gens = cross_df['generator'].astype(str).values[:n_cross]

# Backbone features
with torch.no_grad():
    B = n_test
    t = test_feats[:B].to(device).view(B, model.n_streams, 512) + model.stream_embed
    for layer in model.layers:
        t, _ = layer(t)
    test_bb = model.backbone(t.flatten(1)).cpu()

    B2 = n_cross
    t2 = cross_feats[:B2].to(device).view(B2, model.n_streams, 512) + model.stream_embed
    for layer in model.layers:
        t2, _ = layer(t2)
    cross_bb = model.backbone(t2.flatten(1)).cpu()

rng = np.random.RandomState(42)
idx1 = rng.choice(len(test_bb), min(1500, len(test_bb)), replace=False)
idx2 = rng.choice(len(cross_bb), min(1000, len(cross_bb)), replace=False)

feats_all = torch.cat([test_bb[idx1], cross_bb[idx2]], 0).numpy()
gens_all = np.concatenate([test_gens[idx1], cross_gens[idx2]])
labels_all = np.concatenate([test_labels[:n_test].numpy()[idx1], cross_labels[:n_cross].numpy()[idx2]])

print(f"  Running t-SNE on {len(feats_all)} samples...")
coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(feats_all)

unique_gens = sorted(set(gens_all))
cmap = plt.cm.get_cmap('tab20', len(unique_gens))
gen_color = {g: cmap(i) for i, g in enumerate(unique_gens)}

unseen_gens = {'wildfake_ddim', 'wildfake_other', 'dcgan_unseen', 'fursona_gan', 'waifu_gan'}

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for gen in unique_gens:
    mask = gens_all == gen
    is_u = gen in unseen_gens
    axes[0].scatter(coords[mask, 0], coords[mask, 1], c=[gen_color[gen]], label=gen,
                    marker='*' if is_u else 'o', s=60 if is_u else 15, alpha=0.6)
axes[0].set_title('t-SNE: By Generator', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=7, markerscale=1.5, ncol=2)
axes[0].set_xticks([]); axes[0].set_yticks([])

real_m = labels_all == 0
fake_seen = (labels_all == 1) & ~np.isin(gens_all, list(unseen_gens))
fake_unseen = (labels_all == 1) & np.isin(gens_all, list(unseen_gens))
axes[1].scatter(coords[real_m, 0], coords[real_m, 1], c='#55A868', label='Real', s=15, alpha=0.4)
axes[1].scatter(coords[fake_seen, 0], coords[fake_seen, 1], c='#4C72B0', label='Fake (seen)', s=15, alpha=0.4)
axes[1].scatter(coords[fake_unseen, 0], coords[fake_unseen, 1], c='#C44E52', label='Fake (unseen)', marker='*', s=60, alpha=0.7)
axes[1].set_title('t-SNE: Real/Fake/Unseen', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10, markerscale=1.5)
axes[1].set_xticks([]); axes[1].set_yticks([])

plt.tight_layout()
plt.savefig(OUT_DIR / 'tsne_updated.png', dpi=200)
plt.close()
print(f"  Saved: {OUT_DIR / 'tsne_updated.png'}")
print("\nDone!")
