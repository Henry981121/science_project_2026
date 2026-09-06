"""
v2_fusion.config — 所有超參數的唯一來源
========================================
設計原則（對應 audit 發現 05、20）：
  1. 每個超參數只在這裡定義一次，沒有第二個預設值可以跟它矛盾。
  2. 存進 results.json 的 config 是「這個 dataclass 的實際內容」，
     不是手寫字串 —— 所以結果檔不可能記錄到沒有發生的事。

2026-08-27 變更
---------------
* **GRL 移除。** 不再有 gradient reversal、gen discriminator、lambda_grl。
  原本保留 GRL 是為了做「λ 只乘一次」的乾淨對照（audit 發現 05）；
  既然方向上不採用 domain adaptation，那條路徑連同它的 λ 一起刪掉，
  而不是留著預設 0 —— 留著就還有人會去設它。
* **特徵流換成 dct / clip / dinov2**，維度直接是 backbone 原生輸出，
  不再經過任何隨機初始化的投影層（audit 發現 01 因此消失）。
* **source 類別空間只含 train 裡實際出現的 generator**，
  cross_generator_test 專屬的 generator 標為 -1 並被 ignore_index 吃掉 ——
  不製造永遠收不到樣本的死輸出（audit 發現 08 的同一個病）。
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════
# Generator 標籤表
# ══════════════════════════════════════════════════════════════════════
#
# source head 的類別空間 = train/val 裡實際出現的 10 種 source。
# 這是刻意的：把只出現在 cross_generator_test 的 generator 也編進來的話，
# 那些類別在訓練時永遠收不到樣本，就是 audit 發現 08 的死輸出。

TRAIN_GENERATORS: List[str] = [
    'real',        # 0
    'real_extra',  # 1
    'adm',         # 2
    'glide',       # 3
    'sdv4',        # 4
    'sdv5',        # 5
    'midjourney',  # 6
    'wildfake',    # 7
    'stylegan',    # 8
    'dcgan',       # 9
]

# 只出現在 cross_generator_test。它們是合法名稱（不該 raise），
# 但沒有 source 標籤 —— source id 填 -1，CE 的 ignore_index 會吃掉。
EVAL_ONLY_GENERATORS: List[str] = [
    'wildfake_ddim',
    'wildfake_other',
    'dcgan_unseen',
    'fursona_gan',
    'waifu_gan',
]

GENERATOR_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(TRAIN_GENERATORS)}
SOURCE_ID_TO_NAME: Dict[int, str] = {i: n for n, i in GENERATOR_TO_ID.items()}

KNOWN_GENERATORS: frozenset = frozenset(TRAIN_GENERATORS) | frozenset(EVAL_ONLY_GENERATORS)

# 真圖。這兩個名字是 source head 的類別，但 binary label 為 0。
REAL_GENERATOR_NAMES: List[str] = ['real', 'real_extra']
REAL_IDS = frozenset(GENERATOR_TO_ID[n] for n in REAL_GENERATOR_NAMES)

SOURCE_IGNORE_INDEX = -1

N_SOURCES = len(TRAIN_GENERATORS)   # 10
N_BINARY  = 2


# ══════════════════════════════════════════════════════════════════════
# 特徵流
# ══════════════════════════════════════════════════════════════════════
#
# 2026-08-26 handoff（handoff_dct_clip_dinov2_20260826）。
# 這些維度是各 backbone 的原生輸出，不是被隨機投影壓過的結果：
#   dct     192  = 64 個頻率位置 × 3 個統計量（手工特徵，無 CNN）
#   clip    768  = CLIP ViT-L/14 的 image embedding（CLIP 自己訓練好的 visual
#                  projection 輸出，不是我們加的隨機層）
#   dinov2  1024 = DINOv2 ViT-L/14 的 CLS token
#
# StreamAdapter 的 Linear(in_dim → d_model) 就是唯一的降維層，而它是被訓練的。

STREAMS: List[str] = ['dct', 'clip', 'dinov2']

STREAM_DIMS: Dict[str, int] = {
    'dct':    192,
    'clip':   768,
    'dinov2': 1024,
}

# cache 佈局（handoff package）：
#   {cache}/{stream}/{split}.npy         (N, dim)  float32
#   {cache}/{stream}/{split}.valid.npy   (N,)      bool
#   {cache}/index_{split}.csv            欄位 row,path,generator,is_real,split,valid_all
SPLITS: List[str] = ['train', 'val', 'cross_generator_test']


# ══════════════════════════════════════════════════════════════════════
# 模型設定
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    streams: List[str] = field(default_factory=lambda: list(STREAMS))
    stream_dims: Dict[str, int] = field(default_factory=lambda: dict(STREAM_DIMS))

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2          # stream 之間互相溝通的 self-attention 層數
    ffn_mult: int = 4

    dropout_attn: float = 0.1  # 對應 audit 發現 18：這個值以前傳進去會被吃掉
    dropout_head: float = 0.3

    # audit 發現 14：舊版 randn*0.02，範數約 0.45，相對特徵尺度小 1-2 個數量級。
    # v2 的流身份主要由「每條流各自獨立的 StreamAdapter」承擔，那比一個
    # 加法式 embedding 強得多；這個 embedding 是輔助，保留可調。
    stream_embed_std: float = 0.02

    # EXP2A: keep input streams fixed while changing only the fusion module.
    fusion_mode: str = 'hybrid'  # hybrid, self, cross, concat

    # ── 三軸因子設計（見 fusion_cells.py）───────────────────────────
    # None = 走 legacy 的 fusion_mode 路徑，現有結果完全可重現。
    # 設了 cell_id 就改走 FusionCell，此時 fusion_mode / n_layers 被忽略。
    #
    # cell_id 格式 'o<1|2>.<static|gated>.<feat|dec>'，或參照組 'concat'。
    fusion_cell: Optional[str] = None
    cell_width: int = 64
    # 給了 budget 就用二分搜尋反推 width（覆蓋 cell_width），
    # 這是「參數量對齊」的實作。cell_params 是反推後的實際值，
    # 由 sweep 回填 —— 「我們對齊了參數量」是要能被查核的宣稱。
    cell_budget: Optional[int] = None
    cell_params: Optional[int] = None

    # source head：預測「是哪個 source」的 auxiliary head，正常梯度。
    # 這不是 GRL —— GRL 已移除。預設關閉（LossConfig.lambda_src = 0）。
    use_source_head: bool = True

    n_sources: int = N_SOURCES
    n_binary: int = N_BINARY


@dataclass
class LossConfig:
    """
    Total = CE_binary + lambda_src * CE_source

    沒有 lambda_grl，也沒有 gen loss —— GRL 整條路徑已從模型移除。
    """
    lambda_src: float = 0.0    # 預設關掉。舊版是「意外」開在 0.1 的。


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    clip_grad: float = 1.0
    seed: int = 42             # audit 發現 02：舊版整條路徑 grep manual_seed 零命中
    num_workers: int = 0
    select_metric: str = 'val_auc'   # 用哪個指標挑 best checkpoint


@dataclass
class RunConfig:
    name: str = 'v2_baseline'
    model: ModelConfig = field(default_factory=ModelConfig)
    loss:  LossConfig  = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        """存進 results.json 的內容。直接來自實際物件，沒有手寫字串。"""
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════
# 對照實驗 preset
# ══════════════════════════════════════════════════════════════════════
#
# GRL 移除後只剩一個 λ 可調：source head 的權重。
# 這兩組回答「多任務 auxiliary supervision 有沒有幫助」，只差 λ_src。

def build_presets() -> Dict[str, RunConfig]:
    presets: Dict[str, RunConfig] = {}
    for lam_src in (0.0, 0.1):
        name = f"src{lam_src:g}".replace('.', 'p')
        presets[name] = RunConfig(name=name, loss=LossConfig(lambda_src=lam_src))
    return presets


PRESET_NOTES = {
    'src0':   '乾淨 baseline —— 只有 CE_binary，沒有任何輔助 head',
    'src0p1': '加上 source auxiliary head（正常梯度，非對抗）',
}
