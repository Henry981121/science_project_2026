"""
day4_2b_analysis.py
===================

Level 2 — 2B 的分析半邊：attention vs. ablation 對照（XAI 計畫 Day 4 / 2B）。

讀 day4_2b_ablation.py 產出的 attention_ablation.csv（同檔含 attn_* 與
ablimp_*），產出 attention 與 ablation importance 的對照：

  1. 對照長條圖  <outdir>/attn_vs_ablation_bar.png
       每條流兩根 bar：attention（模型內部加權）vs. ablation（因果 |Δ|），
       兩者皆正規化成「佔總量的比例」以便同尺度比較
  2. 散點圖      <outdir>/attn_vs_ablation_scatter.png
       5 條流的 (attention, ablation) 散點 + Spearman ρ
  3. 摘要表      <outdir>/attn_vs_ablation_summary.csv
       逐流 mean attn / mean ablimp / share / rank

【attention 與 ablation 是不同的東西，別混】
  - attention   = 融合模型內部 cross-attention 的加權，是「模型怎麼看」的描述
  - ablation    = 把某流特徵歸零看 fake-prob 變多少，是「拿掉它結果會不會變」
                  的因果測試（leave-one-out）
ρ 衡量兩者「排名」一致度（−1~+1）。ρ 高才能說 attention 反映因果貢獻；
ρ 低不代表哪個錯，只代表兩者測的不是同一件事。

【純數字分析，不需 GPU / checkpoint，任何機器可跑】

【怎麼用】
    python day4_2b_analysis.py
    python day4_2b_analysis.py <attention_ablation.csv 路徑> --outdir outputs/level2

【相依】pandas, numpy, matplotlib
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STREAMS = ["clip", "fft", "dct", "dire", "noise"]
ATTN_COLS = [f"attn_{s}" for s in STREAMS]
ABL_COLS = [f"ablimp_{s}" for s in STREAMS]

STREAM_LABEL = {"clip": "CLIP", "fft": "FFT", "dct": "DCT",
                "dire": "DIRE", "noise": "Noise"}
STREAM_COLOR = {"clip": "#3b6fb6", "fft": "#e08a3c", "dct": "#4f9d5d",
                "dire": "#c1453b", "noise": "#8a6bb0"}
ATTN_BAR_COLOR = "#3b6fb6"
ABL_BAR_COLOR = "#e08a3c"


# ── Spearman（rank 相關，免 scipy）─────────────────────────────────────
def spearman(x, y) -> float:
    """Spearman ρ = rank 後的 Pearson 相關。"""
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))
    return float(x.rank().corr(y.rank()))


def per_image_spearman(df: pd.DataFrame) -> pd.Series:
    """每張圖各算一個 attention-vs-ablation 的 5 流 Spearman ρ。"""
    ra = df[ATTN_COLS].rank(axis=1)
    rb = df[ABL_COLS].rank(axis=1)
    ca = ra.sub(ra.mean(axis=1), axis=0)
    cb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ca.values * cb.values).sum(axis=1)
    den = np.sqrt((ca.values ** 2).sum(axis=1) * (cb.values ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = num / den
    return pd.Series(rho).dropna()


# ── 載入 ──────────────────────────────────────────────────────────────
def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in ATTN_COLS + ABL_COLS if c not in df.columns]
    if missing:
        sys.exit(f"[2b-analysis] CSV 缺欄位: {missing}\n"
                 "     （需要 day4_2b_ablation.py 產出的 attention_ablation.csv）")
    return df


# ── 畫圖 ──────────────────────────────────────────────────────────────
def plot_grouped_bar(attn_share: pd.Series, abl_share: pd.Series,
                     save_path: Path) -> None:
    x = np.arange(len(STREAMS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - w / 2, [attn_share[s] for s in STREAMS], w,
                label="Attention (model internal weighting)", color=ATTN_BAR_COLOR)
    b2 = ax.bar(x + w / 2, [abl_share[s] for s in STREAMS], w,
                label="Ablation importance (causal |delta|)", color=ABL_BAR_COLOR)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.006,
                    f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([STREAM_LABEL[s] for s in STREAMS])
    ax.set_ylabel("Share of total (normalised)")
    ax.set_title("Level 2 / 2B - Attention vs. ablation importance per stream",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[2b-analysis] saved {save_path}")


def plot_scatter(attn_share: pd.Series, abl_share: pd.Series,
                 rho: float, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = max(attn_share.max(), abl_share.max()) * 1.18
    ax.plot([0, lim], [0, lim], "--", color="0.75", linewidth=1, zorder=1)
    for s in STREAMS:
        ax.scatter(attn_share[s], abl_share[s], s=140, color=STREAM_COLOR[s],
                   edgecolor="black", linewidth=0.8, zorder=3)
        ax.annotate(STREAM_LABEL[s], (attn_share[s], abl_share[s]),
                    textcoords="offset points", xytext=(9, 5), fontsize=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Attention share (model internal weighting)")
    ax.set_ylabel("Ablation importance share (causal)")
    ax.set_title(f"Attention vs. ablation  (Spearman rho = {rho:.2f})",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[2b-analysis] saved {save_path}")


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Level 2 / 2B attention vs ablation 分析")
    ap.add_argument("csv", nargs="?",
                    default="outputs/level2/attention_ablation.csv",
                    help="attention_ablation.csv 路徑")
    ap.add_argument("--outdir", default="outputs/level2", help="輸出資料夾")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"[2b-analysis] 找不到 CSV: {csv_path}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load(csv_path)
    print(f"[2b-analysis] 載入 {len(df)} 筆資料")

    # 逐流平均 + 正規化成 share（兩者同尺度才好比）
    attn_mean = pd.Series({s: df[f"attn_{s}"].mean() for s in STREAMS})
    abl_mean = pd.Series({s: df[f"ablimp_{s}"].mean() for s in STREAMS})
    attn_share = attn_mean / attn_mean.sum()
    abl_share = abl_mean / abl_mean.sum()

    # Spearman ρ
    rho_all = spearman(attn_mean.values, abl_mean.values)
    no_noise = [s for s in STREAMS if s != "noise"]
    rho_no_noise = spearman(attn_mean[no_noise].values, abl_mean[no_noise].values)
    rho_img = per_image_spearman(df)

    # 摘要表
    summary = pd.DataFrame({
        "mean_attn": attn_mean,
        "mean_ablation": abl_mean,
        "attn_share": attn_share,
        "ablation_share": abl_share,
        "attn_rank": attn_mean.rank(ascending=False).astype(int),
        "ablation_rank": abl_mean.rank(ascending=False).astype(int),
    })
    summary.to_csv(outdir / "attn_vs_ablation_summary.csv")

    print("\n=== Per-stream: attention vs ablation ===")
    print(summary.round(4).to_string())
    print("\n=== Spearman rho (attention vs ablation) ===")
    print(f"  5 streams aggregate : {rho_all:.3f}")
    print(f"  excluding Noise     : {rho_no_noise:.3f}")
    print(f"  per-image mean      : {rho_img.mean():.3f}  "
          f"(median {rho_img.median():.3f}, std {rho_img.std():.3f})")
    if "family" in df.columns:
        print("\n=== Per-family aggregate rho ===")
        for fam, g in df.groupby("family"):
            r = spearman(g[ATTN_COLS].mean(axis=0).values,
                         g[ABL_COLS].mean(axis=0).values)
            print(f"  {fam:10s} rho={r:6.3f}  (n={len(g)})")

    # 圖
    plot_grouped_bar(attn_share, abl_share, outdir / "attn_vs_ablation_bar.png")
    plot_scatter(attn_share, abl_share, rho_all,
                 outdir / "attn_vs_ablation_scatter.png")

    print(f"\n[2b-analysis] 完成，全部輸出在 {outdir}/")


if __name__ == "__main__":
    main()
