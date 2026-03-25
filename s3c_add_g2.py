"""
S3C Supplement: Run 3 fusion methods on G2 (unseen) cross_gen_test
Uses cached features from S3A.
"""
import sys, json, torch, torch.nn as nn, torch.optim as optim, numpy as np
from config import OUTPUTS_DIR, FEAT_CACHE_DIR
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import torch.nn.functional as F

FEAT_DIR = FEAT_CACHE_DIR
OUTPUT_DIR = OUTPUTS_DIR / 'exp_c'
STREAMS = ["clip", "fft", "dct", "dire", "noise"]
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH_SIZE, LR = 20, 256, 1e-3

# Load features
def load_split(prefix):
    feats = torch.cat([torch.load(FEAT_DIR/f"{s}_{prefix}_feats.pt", weights_only=False) for s in STREAMS], dim=1)
    labels = torch.load(FEAT_DIR/f"{prefix}_labels.pt", weights_only=False)
    return feats, labels

train_feats, train_labels = load_split('train')
val_feats, val_labels = load_split('val')
cross_feats, cross_labels = load_split('cross_gen_test')
n = len(STREAMS)
print(f"Train: {len(train_feats):,} | Val: {len(val_feats):,} | Cross-gen: {len(cross_feats):,}")

# Models
class ConcatMLP(nn.Module):
    def __init__(self): super().__init__(); self.mlp = nn.Sequential(nn.Linear(n*512,512),nn.ReLU(),nn.Dropout(0.3),nn.Linear(512,2))
    def forward(self,x): return self.mlp(x)

class WeightedFusion(nn.Module):
    def __init__(self): super().__init__(); self.w = nn.Parameter(torch.ones(n)/n); self.head = nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,2))
    def forward(self,x):
        B=x.shape[0]; s=x.view(B,n,512); w=torch.softmax(self.w,0)
        return self.head((s*w.unsqueeze(0).unsqueeze(-1)).sum(1))

class CrossAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(512,8,dropout=0.1,batch_first=True)
        self.n1=nn.LayerNorm(512); self.n2=nn.LayerNorm(512)
        self.ff=nn.Sequential(nn.Linear(512,2048),nn.GELU(),nn.Dropout(0.1),nn.Linear(2048,512),nn.Dropout(0.1))
        self.emb=nn.Parameter(torch.randn(1,n,512)*0.02)
        self.backbone=nn.Sequential(nn.Linear(n*512,1024),nn.LayerNorm(1024),nn.GELU(),nn.Dropout(0.3),nn.Linear(1024,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.3))
        self.head=nn.Linear(512,2)
    def forward(self,x):
        B=x.shape[0]; t=x.view(B,n,512)+self.emb; o,_=self.attn(t,t,t); t=self.n1(t+o); t=self.n2(t+self.ff(t))
        return self.head(self.backbone(t.flatten(1)))

def train_and_eval(name, model):
    model.to(device)
    opt=optim.AdamW(model.parameters(),lr=LR); crit=nn.CrossEntropyLoss()
    loader=DataLoader(TensorDataset(train_feats,train_labels),batch_size=BATCH_SIZE,shuffle=True)
    best_acc=0
    for ep in range(EPOCHS):
        model.train()
        for f,l in loader:
            f,l=f.to(device),l.to(device); opt.zero_grad(); loss=crit(model(f),l); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p=model(val_feats.to(device)).argmax(1).cpu().numpy()
        acc=accuracy_score(val_labels.numpy(),p)*100
        if acc>best_acc: best_acc=acc; best_state=model.state_dict().copy()
    model.load_state_dict(best_state)
    # G1
    model.eval()
    with torch.no_grad():
        lb=model(val_feats.to(device)); g1_acc=accuracy_score(val_labels.numpy(),lb.argmax(1).cpu().numpy())*100
        g1_auc=roc_auc_score(val_labels.numpy(),F.softmax(lb,1)[:,1].cpu().numpy())
    # G2
    with torch.no_grad():
        lb2=model(cross_feats.to(device)); g2_acc=accuracy_score(cross_labels.numpy(),lb2.argmax(1).cpu().numpy())*100
        g2_auc=roc_auc_score(cross_labels.numpy(),F.softmax(lb2,1)[:,1].cpu().numpy())
    print(f"  {name:<20} G1={g1_acc:.2f}% G2={g2_acc:.2f}% (gap={g2_acc-g1_acc:+.2f})")
    return {'g1_acc':round(g1_acc,2),'g1_auc':round(g1_auc,4),'g2_acc':round(g2_acc,2),'g2_auc':round(g2_auc,4),'gap':round(g2_acc-g1_acc,2)}

print(f"\n{'='*60}\n  EXP-C Supplement: G1 vs G2 Comparison\n{'='*60}")
results = {}
for name, cls in [("Concat+MLP", ConcatMLP), ("WeightedFusion", WeightedFusion), ("CrossAttention", CrossAttn)]:
    results[name] = train_and_eval(name, cls())

with open(OUTPUT_DIR / "results_g2.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR / 'results_g2.json'}")
