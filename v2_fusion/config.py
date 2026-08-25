"""
v2_fusion.config — 所有超參數的唯一來源
========================================
設計原則（對應 audit 發現 05、20）：
  1. 每個超參數只在這裡定義一次，沒有第二個預設值可以跟它矛盾。
  2. 存進 results.json 的 config 是「這個 dataclass 的實際內容」，
     不是手寫字串 —— 所以結果檔不可能記錄到沒有發生的事。

舊版的病：
  s3_main_grl.py:18  docstring 說 lambda_src=0
  s3_main_grl.py:221 GRLLoss 預設值 0.0
  s3_main_grl.py:62  實際傳入 0.1   ← 只有這個生效
  s3_main_grl.py:578 結果檔寫死 "curriculum: easy->medium->hard"，但 curriculum 根本停用
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════
# Generator 標籤表
# ══════════════════════════════════════════════════════════════════════

GENERATOR_TO_ID: Dict[str, int] = {
    'adm':            0,
    'glide':          1,
    'sdv4':           2,
    'sdv5':           3,
    'midjourney':     4,
    'wildfake':       5,
    'biggan':         6,
    'vqdm':           7,
    'wukong':         8,
    'firefly':        9,
    'real':          10,
    'real_extra':    11,
    'wildfake_ddim': 12,
    'wildfake_other':13,
    'stylegan':      14,
    'dcgan':         15,
    'dcgan_unseen':  16,
    'fursona_gan':   17,
    'waifu_gan':     18,
}

# 真圖的 source id。這兩個要從 generator discriminator 的類別空間排除。
REAL_GENERATOR_NAMES: List[str] = ['real', 'real_extra']
REAL_IDS = frozenset(GENERATOR_TO_ID[n] for n in REAL_GENERATOR_NAMES)

# 假圖 generator 的「連續」類別空間。
# 對應 audit 發現 08：舊版用 torch.clamp(gen_ids, 0, N_GEN-1) 當 remap，
# 那是截斷不是排除 —— class 10/11 變成永遠收不到樣本的死輸出，
# 而 id 17/18 被壓成 16，跟 dcgan_unseen 共用標籤。
# 這裡改成明確的字典映射，一對一，不可能撞號。
FAKE_GENERATOR_NAMES: List[str] = [
    n for n in GENERATOR_TO_ID if n not in REAL_GENERATOR_NAMES
]
SOURCE_ID_TO_GEN_ID: Dict[int, int] = {
    GENERATOR_TO_ID[name]: i for i, name in enumerate(FAKE_GENERATOR_NAMES)
}
GEN_ID_TO_NAME: Dict[int, str] = {
    i: name for i, name in enumerate(FAKE_GENERATOR_NAMES)
}

N_SOURCES = len(GENERATOR_TO_ID)          # 19
N_GEN     = len(FAKE_GENERATOR_NAMES)     # 17
N_BINARY  = 2

STREAMS: List[str] = ['clip', 'fft', 'dct', 'dire', 'noise']

# 每條流在 cache 裡的輸入維度。
#
# LEGACY（目前隊友機器上的 3.22 cache）：五條都已經被 extractor 內部的
#   投影層壓成 512 —— 但那些投影層是隨機初始化且從未訓練（audit 發現 01），
#   CLIP 丟掉的那一半資訊已經不在檔案裡，模型端救不回來。
#
# RAW（隊友下次重抽特徵時改存的）：存 projection 之前的原始維度，
#   投影層搬進本模型當可訓練的 StreamAdapter（audit 發現 16）。
#   同樣的 backbone、同樣的流、同樣的抽取時間，cache 約大 1.6 倍。
STREAM_DIMS_LEGACY: Dict[str, int] = {
    'clip': 512, 'fft': 512, 'dct': 512, 'dire': 512, 'noise': 512,
}
STREAM_DIMS_RAW: Dict[str, int] = {
    'clip': 1024, 'fft': 512, 'dct': 512, 'dire': 2048, 'noise': 512,
}


# ══════════════════════════════════════════════════════════════════════
# 模型設定
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    streams: List[str] = field(default_factory=lambda: list(STREAMS))
    stream_dims: Dict[str, int] = field(
        default_factory=lambda: dict(STREAM_DIMS_LEGACY))

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

    n_sources: int = N_SOURCES
    n_gen: int = N_GEN
    n_binary: int = N_BINARY


@dataclass
class LossConfig:
    """
    Total = CE_binary
          + lambda_src * CE_source          (作用在 shared 上，正常梯度)
          + 1.0        * CE_gen             (經過 GRL，梯度已被反轉並乘上 -lambda_grl)

    lambda_grl 只出現在 GradientReversal 裡，一次。
    這裡的 gen loss 權重固定 1.0，不可設定 —— 就是為了讓 λ 不可能被乘第二次。
    對應 audit 發現 05：舊版 λ 同時進了 GRL 的 backward 和 loss 權重，
    backbone 實收 λ² = 0.00198，比反方向的 λ_src=0.1 弱 50.6 倍。
    """
    lambda_src: float = 0.0    # 預設關掉。舊版是「意外」開在 0.1 的。
    lambda_grl: float = 0.0    # 預設關掉。要開請顯式指定。

    grl_schedule: str = 'progressive'   # 'progressive' | 'constant'
    grl_gamma: float = 10.0

    def grl_lambda_at(self, epoch: int, total_epochs: int) -> float:
        """Ganin et al. 2015 的 progressive schedule。"""
        if self.lambda_grl == 0.0:
            return 0.0
        if self.grl_schedule == 'constant':
            return self.lambda_grl
        import math
        progress = epoch / max(total_epochs - 1, 1)
        return self.lambda_grl * (2.0 / (1.0 + math.exp(-self.grl_gamma * progress)) - 1.0)


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
# audit 發現 05 指出：先前「GRL 效益 ≈ 0」的結論不成立，因為 GRL 從來沒真的開過。
# 這四組是唯一能回答「GRL 到底有沒有用」的乾淨對照 —— 只差 λ，其餘完全相同。
# 在 cached features 上每組約 6 分鐘。

def build_presets() -> Dict[str, RunConfig]:
    presets: Dict[str, RunConfig] = {}
    for lam_src in (0.0, 0.1):
        for lam_grl in (0.0, 0.05):
            src_tag = f"src{lam_src:g}".replace('.', 'p')
            grl_tag = f"grl{lam_grl:g}".replace('.', 'p')
            name = f"{src_tag}_{grl_tag}"
            presets[name] = RunConfig(
                name=name,
                loss=LossConfig(lambda_src=lam_src, lambda_grl=lam_grl),
            )
    return presets


PRESET_NOTES = {
    'src0_grl0':     '乾淨 baseline —— 只有 CE_binary，沒有任何輔助 head',
    'src0_grl0p05':  'GRL 的真實效益 —— 唯一沒有反向抵消的對抗設定',
    'src0p1_grl0':   'source head 單獨的效益',
    'src0p1_grl0p05':'舊模型的設定（但 λ 只乘一次，所以 GRL 真的有開）',
}
