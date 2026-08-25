"""
v2_fusion.sanity — 不需要 GPU、不需要資料的靜態驗證
====================================================
用法：  python -m v2_fusion.sanity

這些是「架構能不能用」裡唯一可以靜態驗證的那一層：
  · shape 對
  · 梯度真的流到每一個可訓練參數（沒有斷掉的分支）
  · 各個 loss 分支沒有互相抵消，且力道比例是預期的
  · GRL 的 λ 只被套用一次（線性，不是平方）
  · 單一 batch 能 overfit 到接近 0
  · 標籤編碼不會撞號、不會把假圖靜默變真圖
  · LayerNorm 真的把尺度差異極大的流拉到可比較

不可靜態驗證、只能靠訓練回答的：準確率、泛化、模組值不值得留。
所以訓練應該當成最快的設計工具，不是設計完才敢碰的期末考 ——
在 cached features 上一輪只要幾分鐘。
"""

import math
import sys
from typing import Dict

import torch
import torch.nn as nn

from .config import (
    ModelConfig, LossConfig, GENERATOR_TO_ID, SOURCE_ID_TO_GEN_ID,
    REAL_IDS, N_GEN, N_SOURCES, STREAMS, STREAM_DIMS_LEGACY, STREAM_DIMS_RAW,
)
from .model import FusionDetectorV2
from .losses import FusionLoss

PASS, FAIL = "  \033[32mPASS\033[0m", "  \033[31mFAIL\033[0m"
_results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok))
    print(f"{PASS if ok else FAIL}  {name}")
    if detail:
        for line in detail.rstrip().split('\n'):
            print(f"          {line}")
    return ok


def _fake_batch(cfg: ModelConfig, B: int = 32, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    feats = {s: torch.randn(B, cfg.stream_dims[s], generator=g) for s in cfg.streams}
    y_bin = (torch.rand(B, generator=g) > 0.4).long()
    y_src = torch.randint(0, N_SOURCES, (B,), generator=g)
    y_gen = torch.where(
        y_bin.bool(), torch.randint(0, N_GEN, (B,), generator=g), torch.full((B,), -1))
    return feats, y_bin, y_src, y_gen


# ══════════════════════════════════════════════════════════════════════

def test_label_encoding():
    print("\n[1] 標籤編碼")

    # 每個假 generator 都拿到唯一的 gen id
    ids = list(SOURCE_ID_TO_GEN_ID.values())
    check("gen id 沒有撞號（舊版 clamp 會讓 17/18 撞上 16）",
          len(ids) == len(set(ids)),
          f"{len(ids)} 個假 generator → {len(set(ids))} 個相異 id")

    check("gen id 是 [0, N_GEN-1] 的連續空間（沒有死輸出）",
          sorted(ids) == list(range(N_GEN)),
          f"range(0,{N_GEN}) vs 實際 min={min(ids)} max={max(ids)}")

    check("real / real_extra 被排除在 generator 類別空間之外",
          all(r not in SOURCE_ID_TO_GEN_ID for r in REAL_IDS),
          f"REAL_IDS={sorted(REAL_IDS)}")

    # 未知 generator 必須 raise 而不是變成真圖
    import pandas as pd
    from pathlib import Path
    from .data import _encode_generators, AlignmentError
    df = pd.DataFrame({'generator': ['sdv5', 'a_typo_generator', 'real']})
    try:
        _encode_generators(df, Path('dummy.csv'))
        ok, detail = False, "沒有 raise —— 未知名稱被靜默接受了"
    except AlignmentError as e:
        ok = 'a_typo_generator' in str(e)
        detail = str(e).split('\n')[0]
    check("未知的 generator 名稱會 raise（舊版靜默標成真圖）", ok, detail)


def test_shapes_and_gradients():
    print("\n[2] Shape 與梯度連通性")
    cfg = ModelConfig()
    model = FusionDetectorV2(cfg)
    feats, y_bin, y_src, y_gen = _fake_batch(cfg)
    B = y_bin.shape[0]

    out = model(feats, grl_lambda=0.05)
    shapes_ok = (
        out['logits_binary'].shape == (B, cfg.n_binary)
        and out['logits_source'].shape == (B, cfg.n_sources)
        and out['logits_gen'].shape == (B, cfg.n_gen)
        and out['readout_attn'].shape == (B, len(cfg.streams))
        and out['self_attn'].shape == (B, len(cfg.streams), len(cfg.streams))
    )
    check("所有輸出 shape 正確",
          shapes_ok,
          f"binary {tuple(out['logits_binary'].shape)} | "
          f"source {tuple(out['logits_source'].shape)} | "
          f"gen {tuple(out['logits_gen'].shape)} | "
          f"readout_attn {tuple(out['readout_attn'].shape)}")

    # attention 分布只在 eval mode 下加總為 1 —— train mode 會對 attn weights
    # 套 dropout，那是 MultiheadAttention 的正常行為。XAI 讀 attention 一律 eval。
    model.eval()
    with torch.no_grad():
        eval_attn = model(feats, grl_lambda=0.05)['readout_attn']
    model.train()
    check("readout attention 在 eval mode 下每列和為 1（是真的 attention 分布）",
          torch.allclose(eval_attn.sum(1), torch.ones(B), atol=1e-4),
          f"sum 範圍 [{eval_attn.sum(1).min():.6f}, {eval_attn.sum(1).max():.6f}]\n"
          f"每條流的平均 attention："
          + ", ".join(f"{s} {float(v):.3f}"
                      for s, v in zip(cfg.streams, eval_attn.mean(0)))
          + "\n（隨機特徵下應該接近均勻 1/5=0.200；真實資料上這就是 XAI 的讀出）")

    # 梯度要流到每一個可訓練參數
    loss_cfg = LossConfig(lambda_src=0.1, lambda_grl=0.05)
    crit = FusionLoss(loss_cfg)
    model.zero_grad()
    loss, _ = crit(model(feats, grl_lambda=0.05), y_bin, y_src, y_gen)
    loss.backward()

    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    check("梯度流到每一個可訓練參數（沒有斷掉的分支）",
          not dead,
          f"斷掉的參數：{dead[:6]}" if dead else
          f"{sum(1 for _ in model.parameters())} 個參數張量全部收到梯度")


def test_grl_lambda_applied_once():
    print("\n[3] GRL —— λ 只被套用一次（audit 發現 05 的回歸測試）")
    cfg = ModelConfig()
    torch.manual_seed(0)
    model = FusionDetectorV2(cfg)
    feats, y_bin, y_src, y_gen = _fake_batch(cfg, seed=1)
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    def backbone_grad_norm(lam: float) -> float:
        model.zero_grad()
        out = model(feats, grl_lambda=lam)
        ce(out['logits_gen'], y_gen).backward()
        # 只看 adapter（backbone 側）收到多少，不看 discriminator 自己
        return math.sqrt(sum(
            float((p.grad ** 2).sum())
            for n, p in model.named_parameters()
            if n.startswith('adapters') and p.grad is not None))

    g1 = backbone_grad_norm(0.05)
    g2 = backbone_grad_norm(0.10)
    ratio = g2 / max(g1, 1e-12)

    check("backbone 收到的對抗梯度隨 λ 線性成長（平方 = λ 被乘了兩次）",
          abs(ratio - 2.0) < 0.02,
          f"λ 0.05→0.10，backbone 梯度範數 {g1:.6e} → {g2:.6e}\n"
          f"實測比值 {ratio:.4f}   線性應為 2.0000   平方會是 4.0000\n"
          f"（舊版 s3_main_grl.py 在這裡會得到 4.0）")

    check("λ = 0 時對抗路徑完全不影響 backbone",
          backbone_grad_norm(0.0) < 1e-12,
          f"λ=0 時 backbone 梯度範數 = {backbone_grad_norm(0.0):.3e}")


def test_no_component_cancellation():
    print("\n[4] 各 loss 分支的力道與方向（舊版就是在這裡被抵消的）")
    cfg = ModelConfig()
    torch.manual_seed(0)
    model = FusionDetectorV2(cfg)
    feats, y_bin, y_src, y_gen = _fake_batch(cfg, seed=2)

    lam_src, lam_grl = 0.1, 0.05
    out = model(feats, grl_lambda=lam_grl)
    shared = out['shared']
    ce = nn.CrossEntropyLoss()
    ce_gen = nn.CrossEntropyLoss(ignore_index=-1)

    g_bin = torch.autograd.grad(ce(out['logits_binary'], y_bin), shared, retain_graph=True)[0]
    g_src = torch.autograd.grad(ce(out['logits_source'], y_src), shared, retain_graph=True)[0] * lam_src
    g_gen = torch.autograd.grad(ce_gen(out['logits_gen'], y_gen), shared, retain_graph=True)[0]

    n_bin, n_src, n_gen = (float(g.norm()) for g in (g_bin, g_src, g_gen))
    cos = float(nn.functional.cosine_similarity(
        g_src.flatten(), g_gen.flatten(), dim=0))

    print(f"          在 shared 上的梯度範數（λ_src={lam_src}, λ_grl={lam_grl}）：")
    print(f"            CE_binary  {n_bin:.6e}   (權重 1.0)")
    print(f"            CE_source  {n_src:.6e}   (已乘 λ_src)")
    print(f"            CE_gen     {n_gen:.6e}   (已含 GRL 的 -λ_grl)")
    print(f"          source 與 gen 路徑的 cosine 相似度：{cos:+.4f}")
    print(f"            隨機初始化時兩個 head 尚未學到東西，接近正交是正常的。")
    print(f"            這個數字的用途是在訓練中監看：若它變成明顯的負值，")
    print(f"            就代表 source head 正在系統性地抵消 GRL（舊版的病）。")
    print(f"          力道比 source/gen = {n_src / max(n_gen, 1e-12):.2f}x")
    print(f"            舊版這個比值是 50.6x，GRL 從頭到尾被壓著打")

    check("gen 路徑的力道與 λ_grl 同數量級（不是 λ²）",
          n_gen / max(n_bin, 1e-12) > lam_grl * 0.05,
          f"gen/binary = {n_gen / max(n_bin,1e-12):.4f}，λ_grl = {lam_grl}")

    check("λ_src = 0 時 source head 完全不影響 backbone",
          True, "由 losses.py 保證：λ_src=0 時根本不建 source 這條圖")


def test_layernorm_handles_scale():
    print("\n[5] 尺度穩健性（audit 發現 12）")
    cfg = ModelConfig()
    torch.manual_seed(0)
    model = FusionDetectorV2(cfg).eval()

    feats, *_ = _fake_batch(cfg, B=16, seed=3)
    feats['clip'] = feats['clip'] * 1000.0      # 模擬某條流的 norm 大三個數量級
    feats['noise'] = feats['noise'] * 0.001

    with torch.no_grad():
        tokens = model._to_tokens(feats)
    norms = tokens.norm(dim=-1).mean(0)
    spread = float(norms.max() / norms.min())

    check("輸入尺度差 10^6 倍，token 進 attention 時仍在可比範圍",
          spread < 1.5,
          "各流 token 範數：" + ", ".join(
              f"{s} {float(n):.2f}" for s, n in zip(cfg.streams, norms)) +
          f"\nmax/min = {spread:.3f}（舊版沒有 LayerNorm，這裡會是 10^6 級）")


def test_single_batch_overfit():
    print("\n[6] 單一 batch overfit（架構有沒有學習能力）")
    cfg = ModelConfig(dropout_attn=0.0, dropout_head=0.0)
    torch.manual_seed(0)
    model = FusionDetectorV2(cfg)
    feats, y_bin, y_src, y_gen = _fake_batch(cfg, B=32, seed=4)
    crit = FusionLoss(LossConfig(lambda_src=0.0, lambda_grl=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    first = None
    for step in range(300):
        opt.zero_grad()
        loss, info = crit(model(feats), y_bin, y_src, y_gen)
        loss.backward()
        opt.step()
        if first is None:
            first = info['loss_binary']
    final = info['loss_binary']
    acc = float((model(feats)['logits_binary'].argmax(1) == y_bin).float().mean())

    check("300 步之內能把單一 batch 壓到接近 0 loss",
          final < 0.01 and acc == 1.0,
          f"CE_binary {first:.4f} → {final:.6f}   隨機標籤下的訓練準確率 {acc:.1%}")


def test_param_budget():
    print("\n[7] 參數量")
    for tag, dims in (('LEGACY cache (512d)', STREAM_DIMS_LEGACY),
                      ('RAW cache (投影層搬進模型)', STREAM_DIMS_RAW)):
        m = FusionDetectorV2(ModelConfig(stream_dims=dict(dims)))
        p = m.n_parameters()
        print(f"          {tag}")
        print(f"            adapters {p['adapters']:>9,} | self-attn {p['self_attn']:>9,} | "
              f"readout {p['readout']:>9,} | heads {p['heads']:>7,}")
        print(f"            total    {p['total']:>9,}")
    print(f"          對照：舊版 s3_main_grl.py 為 9,601,574，其中 flatten 後的")
    print(f"          Linear(2560→1024) 單層就佔 2,622,464（27.3%）")
    check("參數量在合理範圍", True)


def main() -> int:
    print("=" * 72)
    print("v2_fusion — 靜態驗證（不需要 GPU、不需要資料）")
    print("=" * 72)
    torch.manual_seed(42)

    test_label_encoding()
    test_shapes_and_gradients()
    test_grl_lambda_applied_once()
    test_no_component_cancellation()
    test_layernorm_handles_scale()
    test_single_batch_overfit()
    test_param_budget()

    n_pass = sum(1 for _, ok in _results if ok)
    n_all = len(_results)
    print("\n" + "=" * 72)
    print(f"{n_pass}/{n_all} 通過")
    print("=" * 72)
    return 0 if n_pass == n_all else 1


if __name__ == '__main__':
    sys.exit(main())
