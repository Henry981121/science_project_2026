"""
day4_level2.py
==============

Level 2 — 流級 attention 分布視覺化（XAI 計畫 Day 4）。

讀 pre-flight Risk B 產出的 preflight_attn.csv（每張測試圖 5 條流的
cross-attention 權重 + generator + family），產出：

  1. 主圖   <outdir>/attention_by_generator.png
            —— 按「具體 generator」分組的 stacked bar，展示每個 generator
               啟動不同流的差異化模式（XAI 計畫 2C 主圖）
  2. 附圖   <outdir>/attention_by_family.png
            —— 按家族（Real / GAN / Diffusion）分組（XAI 計畫 2C 附圖）
  3. 表格   <outdir>/attention_by_generator.csv
            <outdir>/attention_by_family.csv
            —— 聚合後的平均 attention，給論文正文引數字用

【不含 ablation cross-check】
XAI 計畫 2B 的 attention-vs-ablation Spearman ρ 需要逐流 zero-out 重跑
fusion model，那要在有 checkpoint 的機器上跑，不在這支腳本範圍。
這支只處理「已存在的 attention CSV → 畫圖」，任何機器都能跑。

【怎麼用】
    python day4_level2.py                            # 預設讀 outputs/preflight_attn.csv
    python day4_level2.py <preflight_attn.csv 路徑>
    python day4_level2.py <csv 路徑> --outdir outputs/level2

【相依】pandas, matplotlib（見 requirements.txt）
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # 不需 display，直接存檔
import matplotlib.pyplot as plt


# ── 設定 ──────────────────────────────────────────────────────────────
STREAMS = ["clip", "fft", "dct", "dire", "noise"]
ATTN_COLS = [f"attn_{s}" for s in STREAMS]

STREAM_LABEL = {
    "clip": "CLIP", "fft": "FFT", "dct": "DCT", "dire": "DIRE", "noise": "Noise",
}
# 5 條流固定配色（stacked bar 各段與 legend 一致）
STREAM_COLOR = {
    "clip":  "#3b6fb6",   # blue
    "fft":   "#e08a3c",   # orange
    "dct":   "#4f9d5d",   # green
    "dire":  "#c1453b",   # red
    "noise": "#8a6bb0",   # purple
}

# generator 顯示順序：依家族分群（Real → GAN → Diffusion）
GENERATOR_ORDER = [
    "real", "real_extra",                                       # Real
    "dcgan", "stylegan",                                        # GAN
    "adm", "glide", "midjourney", "sdv4", "sdv5", "wildfake",    # Diffusion
]
FAMILY_ORDER = ["Real", "GAN", "Diffusion"]


# ── 載入 / 聚合 ────────────────────────────────────────────────────────
def load_attn_csv(csv_path: Path) -> pd.DataFrame:
    """讀 preflight_attn.csv，檢查欄位，只留 split==test。"""
    df = pd.read_csv(csv_path)
    need = ATTN_COLS + ["generator", "family", "split"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"[day4] CSV 缺欄位: {missing}")
    df = df[df["split"] == "test"].copy()
    if df.empty:
        sys.exit("[day4] CSV 沒有 split==test 的資料")
    return df


def aggregate(df: pd.DataFrame, group_col: str, order: list) -> pd.DataFrame:
    """對 group_col 分組，算 5 條流 attention 的平均（每張圖等權），附 n 欄。"""
    g = df.groupby(group_col)
    table = g[ATTN_COLS].mean()
    table["n"] = g.size()
    # 依指定順序排；CSV 裡有但 order 沒列到的接在後面，不丟資料
    present = [x for x in order if x in table.index]
    extra = [x for x in table.index if x not in order]
    return table.loc[present + extra]


# ── 畫圖 ──────────────────────────────────────────────────────────────
def plot_stacked(
    table: pd.DataFrame,
    title: str,
    save_path: Path,
    family_of: Optional[dict] = None,
) -> None:
    """畫 5 條流的 stacked bar。family_of 有給時，在家族交界畫分隔線並標家族名。"""
    groups = list(table.index)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(groups)), 5.0))

    bottom = [0.0] * len(groups)
    for s in STREAMS:
        vals = table[f"attn_{s}"].tolist()
        ax.bar(groups, vals, bottom=bottom, label=STREAM_LABEL[s],
               color=STREAM_COLOR[s], edgecolor="white", linewidth=0.5)
        # 夠大的段在中央標數值
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 0.05:
                ax.text(i, b + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white")
        bottom = [b + v for b, v in zip(bottom, vals)]

    # 家族分隔線 + 家族名（只在 per-generator 圖）
    if family_of is not None:
        fams = [family_of.get(g, "") for g in groups]
        start = 0
        for i in range(1, len(groups) + 1):
            if i == len(groups) or fams[i] != fams[start]:
                if start > 0:
                    ax.axvline(start - 0.5, color="0.7", linestyle="--", linewidth=1)
                ax.text((start + i - 1) / 2, 1.04, fams[start], ha="center",
                        va="bottom", fontsize=10, fontweight="bold", color="0.3")
                start = i

    ax.set_ylabel("Mean cross-attention weight")
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=24)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n(n={int(table.loc[g, 'n'])})" for g in groups],
                       rotation=30, ha="right", fontsize=9)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[day4] saved {save_path}")


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Level 2 attention 分布視覺化")
    ap.add_argument("csv", nargs="?", default="outputs/preflight_attn.csv",
                    help="preflight_attn.csv 路徑（預設 outputs/preflight_attn.csv）")
    ap.add_argument("--outdir", default="outputs/level2",
                    help="輸出資料夾（預設 outputs/level2）")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"[day4] 找不到 CSV: {csv_path}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_attn_csv(csv_path)
    print(f"[day4] 載入 {len(df)} 筆 test 資料")

    gen_tbl = aggregate(df, "generator", GENERATOR_ORDER)
    fam_tbl = aggregate(df, "family", FAMILY_ORDER)

    # console 印聚合表（論文正文引數字用）
    print("\n=== Per-generator mean attention ===")
    print(gen_tbl[ATTN_COLS + ["n"]].round(4).to_string())
    print("\n=== Per-family mean attention ===")
    print(fam_tbl[ATTN_COLS + ["n"]].round(4).to_string())

    # 存聚合表
    gen_tbl.round(6).to_csv(outdir / "attention_by_generator.csv")
    fam_tbl.round(6).to_csv(outdir / "attention_by_family.csv")
    print(f"\n[day4] 聚合表存到 {outdir}/")

    # 畫圖
    family_of = (df.drop_duplicates("generator")
                   .set_index("generator")["family"].to_dict())
    plot_stacked(gen_tbl, "Level 2 - Cross-attention by generator",
                 outdir / "attention_by_generator.png", family_of=family_of)
    plot_stacked(fam_tbl, "Level 2 - Cross-attention by family",
                 outdir / "attention_by_family.png")

    print(f"[day4] 完成，全部輸出在 {outdir}/")


if __name__ == "__main__":
    main()
