"""
config.py — 專案路徑設定
=========================
請在下方設定你的 BASE_DIR（專案根目錄）。
其他所有腳本都從這裡讀取路徑，不需要個別修改。
"""

from pathlib import Path

# ── 請修改這一行，改成你自己的專案根目錄 ──────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
# 如果你的 data/ 和 outputs/ 不在同一個目錄，可以分開設定：
# BASE_DIR = Path('/your/custom/path')

# ── 衍生路徑（通常不需要修改）────────────────────────────────────────────
DATA_DIR       = BASE_DIR / 'data'
SPLITS_DIR     = DATA_DIR / 'splits'
OUTPUTS_DIR    = BASE_DIR / 'outputs'
FEAT_CACHE_DIR = OUTPUTS_DIR / 'exp_a' / 'features'

# CSV splits
TRAIN_CSV  = SPLITS_DIR / 'train.csv'
VAL_CSV    = SPLITS_DIR / 'val.csv'
TEST_CSV   = SPLITS_DIR / 'test.csv'
CROSS_CSV  = SPLITS_DIR / 'cross_generator_test.csv'

# Model checkpoints
MODEL_GRL_PATH  = OUTPUTS_DIR / 'main_grl' / 'best_model.pth'
MODEL_MAIN_PATH = OUTPUTS_DIR / 'main_grl' / 'best_model.pth'
