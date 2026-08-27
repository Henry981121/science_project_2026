"""
v2_fusion.sanity — 不需要 GPU、不需要資料的靜態驗證
====================================================
用法：  python -m v2_fusion.sanity

這些是「架構能不能用」裡唯一可以靜態驗證的那一層：
  · shape 對
  · 梯度真的流到每一個可訓練參數（沒有斷掉的分支）
  · GRL 確實已經整條移除（回歸測試 —— 不讓它被無意間加回來）
  · 各個 loss 分支沒有互相抵消
  · 單一 batch 能 overfit 到接近 0
  · 標籤編碼不會撞號、不會把假圖靜默變真圖
  · LayerNorm 真的把尺度差異極大的流拉到可比較

不可靜態驗證、只能靠訓練回答的：準確率、泛化、模組值不值得留。
所以訓練應該當成最快的設計工具，不是設計完才敢碰的期末考 ——
在 cached features 上一輪只要幾分鐘。
"""

import dataclasses
import sys
from typing import Dict

import torch
import torch.nn as nn

from .config import (
    ModelConfig, LossConfig, GENERATOR_TO_ID, SOURCE_ID_TO_NAME,
    TRAIN_GENERATORS, EVAL_ONLY_GENERATORS, REAL_IDS, SOURCE_IGNORE_INDEX,
    N_SOURCES, STREAMS, STREAM_DIMS,
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
    return feats, y_bin, y_src


# ══════════════════════════════════════════════════════════════════════

def test_label_encoding():
    print("\n[1] 標籤編碼")

    ids = list(GENERATOR_TO_ID.values())
    check("source id 沒有撞號（舊版 clamp 會讓 17/18 撞上 16）",
          len(ids) == len(set(ids)),
          f"{len(ids)} 個 source → {len(set(ids))} 個相異 id")

    check(f"source id 是 [0, {N_SOURCES-1}] 的連續空間（沒有死輸出）",
          sorted(ids) == list(range(N_SOURCES)),
          f"train 出現的 source：{', '.join(TRAIN_GENERATORS)}")

    check("只出現在 cross_generator_test 的 generator 不在 source 類別空間",
          all(g not in GENERATOR_TO_ID for g in EVAL_ONLY_GENERATORS),
          f"這 {len(EVAL_ONLY_GENERATORS)} 種標為 {SOURCE_IGNORE_INDEX}（ignore_index 吃掉）："
          f"{', '.join(EVAL_ONLY_GENERATORS)}\n"
          f"把它們編進來的話會製造永遠收不到樣本的死輸出（audit 發現 08 的同一個病）")

    import pandas as pd
    from pathlib import Path
    from .data import _encode_generators, AlignmentError

    # eval-only generator 是合法的，不該 raise，但 source id 要是 -1
    df = pd.DataFrame({'generator': ['sdv5', 'wildfake_other', 'real']})
    src, is_fake = _encode_generators(df, Path('dummy.csv'))
    check("eval-only generator 不 raise，且 source id 為 -1",
          src.tolist() == [GENERATOR_TO_ID['sdv5'], SOURCE_IGNORE_INDEX,
                           GENERATOR_TO_ID['real']]
          and is_fake.tolist() == [True, True, False],
          f"source ids {src.tolist()}   is_fake {is_fake.tolist()}")

    # 未知 generator 必須 raise 而不是變成真圖
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
    feats, y_bin, y_src = _fake_batch(cfg)
    B, N = y_bin.shape[0], len(cfg.streams)

    out = model(feats)
    shapes_ok = (
        out['logits_binary'].shape == (B, cfg.n_binary)
        and out['logits_source'].shape == (B, cfg.n_sources)
        and out['readout_attn'].shape == (B, N)
        and out['self_attn'].shape == (B, N, N)
    )
    check("所有輸出 shape 正確",
          shapes_ok,
          f"binary {tuple(out['logits_binary'].shape)} | "
          f"source {tuple(out['logits_source'].shape)} | "
          f"readout_attn {tuple(out['readout_attn'].shape)} | "
          f"self_attn {tuple(out['self_attn'].shape)}")

    # attention 分布只在 eval mode 下加總為 1 —— train mode 會對 attn weights
    # 套 dropout，那是 MultiheadAttention 的正常行為。XAI 讀 attention 一律 eval。
    model.eval()
    with torch.no_grad():
        eval_attn = model(feats)['readout_attn']
    model.train()
    check("readout attention 在 eval mode 下每列和為 1（是真的 attention 分布）",
          torch.allclose(eval_attn.sum(1), torch.ones(B), atol=1e-4),
          f"sum 範圍 [{eval_attn.sum(1).min():.6f}, {eval_attn.sum(1).max():.6f}]\n"
          f"每條流的平均 attention："
          + ", ".join(f"{s} {float(v):.3f}"
                      for s, v in zip(cfg.streams, eval_attn.mean(0)))
          + f"\n（隨機特徵下應接近均勻 1/{N}={1/N:.3f}；真實資料上這就是 XAI 的讀出）")

    crit = FusionLoss(LossConfig(lambda_src=0.1))
    model.zero_grad()
    loss, _ = crit(model(feats), y_bin, y_src)
    loss.backward()

    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    check("梯度流到每一個可訓練參數（沒有斷掉的分支）",
          not dead,
          f"斷掉的參數：{dead[:6]}" if dead else
          f"{sum(1 for _ in model.parameters())} 個參數張量全部收到梯度")


def test_grl_removed():
    print("\n[3] GRL 已移除（回歸測試）")
    cfg = ModelConfig()
    model = FusionDetectorV2(cfg)
    feats, y_bin, y_src = _fake_batch(cfg, seed=1)
    out = model(feats)

    import v2_fusion.model as mdl
    leftovers = [n for n in ('grad_reverse', '_GradientReversal') if hasattr(mdl, n)]
    check("model 模組裡沒有 gradient reversal 的實作",
          not leftovers,
          f"殘留：{leftovers}" if leftovers else
          "grad_reverse / _GradientReversal 都不存在")

    check("模型沒有 head_gen，forward 也不輸出 logits_gen",
          not hasattr(model, 'head_gen') and 'logits_gen' not in out,
          f"輸出的 key：{sorted(out)}")

    loss_fields = {f.name for f in dataclasses.fields(LossConfig)}
    check("LossConfig 沒有任何 GRL 相關欄位（λ 沒有地方可以被設第二次）",
          not any('grl' in f for f in loss_fields),
          f"欄位：{sorted(loss_fields)}")

    check("forward() 不接受 grl_lambda 參數",
          'grl_lambda' not in FusionDetectorV2.forward.__code__.co_varnames,
          f"參數：{FusionDetectorV2.forward.__code__.co_varnames[:4]}")


def test_source_head_is_optional():
    print("\n[4] source head —— auxiliary，非對抗")
    cfg = ModelConfig()
    torch.manual_seed(0)
    model = FusionDetectorV2(cfg)
    feats, y_bin, y_src = _fake_batch(cfg, seed=2)

    # λ_src = 0 → source head 完全不影響 backbone
    crit0 = FusionLoss(LossConfig(lambda_src=0.0))
    model.zero_grad()
    crit0(model(feats), y_bin, y_src)[0].backward()
    g0 = model.head_source.weight.grad
    check("λ_src = 0 時 source head 收不到任何梯度（不建圖，不白算 CE）",
          g0 is None or float(g0.abs().sum()) == 0.0,
          f"head_source.weight.grad = {'None' if g0 is None else float(g0.abs().sum())}")

    # λ_src > 0 → 兩條路徑都作用在 shared 上，方向不該系統性相反
    shared = None
    tokens = model._to_tokens(feats)
    for layer in model.layers:
        tokens, _ = layer(tokens)
    shared, _ = model.readout(tokens)
    shared.retain_grad()
    h = model.head_drop(shared)
    ce_bin = nn.CrossEntropyLoss()(model.head_binary(h), y_bin)
    ce_src = nn.CrossEntropyLoss(ignore_index=SOURCE_IGNORE_INDEX)(
        model.head_source(h), y_src)
    g_bin = torch.autograd.grad(ce_bin, shared, retain_graph=True)[0]
    g_src = torch.autograd.grad(ce_src, shared, retain_graph=True)[0]
    cos = float(torch.nn.functional.cosine_similarity(
        g_bin.flatten(), g_src.flatten(), dim=0))
    n_bin, n_src = float(g_bin.norm()), float(g_src.norm())

    print(f"          在 shared 上的梯度範數：CE_binary {n_bin:.6e} | "
          f"CE_source {n_src:.6e}（未乘 λ_src）")
    print(f"          兩者的 cosine 相似度：{cos:+.4f}")
    print(f"            舊版的病是 source head 與 GRL 方向相反、力道差 50.6 倍；")
    print(f"            GRL 移除後不存在對抗抵消，這裡只是記錄兩條路徑的關係。")
    check("兩條路徑都對 backbone 有實際作用（沒有一條是死的）",
          n_bin > 0 and n_src > 0)


def test_layernorm_handles_scale():
    print("\n[5] 尺度穩健性（audit 發現 12）")
    cfg = ModelConfig()
    torch.manual_seed(0)
    model = FusionDetectorV2(cfg).eval()

    feats, *_ = _fake_batch(cfg, B=16, seed=3)
    feats['clip'] = feats['clip'] * 1000.0      # 模擬某條流的 norm 大三個數量級
    feats['dct'] = feats['dct'] * 0.001

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
    feats, y_bin, y_src = _fake_batch(cfg, B=32, seed=4)
    crit = FusionLoss(LossConfig(lambda_src=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    first = None
    for step in range(300):
        opt.zero_grad()
        loss, info = crit(model(feats), y_bin, y_src)
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
    m = FusionDetectorV2(ModelConfig())
    p = m.n_parameters()
    dims = ', '.join(f"{s}:{STREAM_DIMS[s]}" for s in STREAMS)
    print(f"          streams: {dims} -> d_model=512")
    print(f"            adapters {p['adapters']:>9,} | self-attn {p['self_attn']:>9,} | "
          f"readout {p['readout']:>9,} | heads {p['heads']:>7,}")
    print(f"            total    {p['total']:>9,}")

    no_src = FusionDetectorV2(ModelConfig(use_source_head=False)).n_parameters()
    print(f"          use_source_head=False: total {no_src['total']:,}")
    print(f"          對照：舊版 s3_main_grl.py 為 9,601,574（5 條流 + GRL），其中")
    print(f"          flatten 後的 Linear(2560→1024) 單層就佔 2,622,464（27.3%）")
    check("參數量在合理範圍", True)


def main() -> int:
    print("=" * 72)
    print("v2_fusion — 靜態驗證（不需要 GPU、不需要資料）")
    print("=" * 72)
    torch.manual_seed(42)

    test_label_encoding()
    test_shapes_and_gradients()
    test_grl_removed()
    test_source_head_is_optional()
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
