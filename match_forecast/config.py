"""
Project-wide configuration for match_forecast package.

Defines:
  - Filesystem paths (data, models, reports)
  - Random seed and data split ratios
  - Feature conventions (suffixes, categorical columns)
  - Player-processing settings (position mapping, metadata columns)
  - Available model names
  - Logging integration with tqdm
"""
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env, if present
load_dotenv()

# ─── Project paths ─────────────────────────────────────────────────────────────
PROJ_ROOT           = Path(__file__).resolve().parents[1]
DATA_DIR            = PROJ_ROOT / "data"
RAW_DATA_DIR        = DATA_DIR / "raw"
INTERIM_DATA_DIR    = DATA_DIR / "interim"
PROCESSED_DATA_DIR  = DATA_DIR / "processed"
EXTERNAL_DATA_DIR   = DATA_DIR / "external"

MODELS_DIR          = PROJ_ROOT / "models"
CONFIG_DIR          = PROJ_ROOT / "config"
REPORTS_DIR         = PROJ_ROOT / "reports"
FIGURES_DIR         = REPORTS_DIR / "figures"

# ─── Split & seed ────────────────────────────────────────────────────────────────
RANDOM_STATE        = 42
TRAIN_SIZE          = 0.8
VALIDATION_SIZE     = 0.2
TEST_SIZE           = 0.2

# ─── Feature conventions ─────────────────────────────────────────────────────────
NUMERIC_METRIC      = "_average"  # default suffix for numeric features
CATEGORICAL_COLUMNS = ["POSITION"]

# ─── Player processing settings ──────────────────────────────────────────────────
# Map raw player POSITION values to broader groups
POSITION_MAP        = {
    'attacker':   'offensive',
    'midfielder': 'offensive',
    'defender':   'defender',
    'goalkeeper': 'goalkeeper'
}
# Metadata columns to drop
META_COLS_TEAMS     = ['LEAGUE', 'TEAM_NAME']
META_COLS_PLAYERS   = ['LEAGUE', 'TEAM_NAME', 'PLAYER_NAME']

# ─── Model registry ─────────────────────────────────────────────────────────────
# True  : apply StandardScaler -> PCA -> StandardScaler before training
# False : no scaling/PCA needed
PREPROCESSING_REQUIRED = {
    "rf":       False,
    "xt":       False,
    "xgb":      False,
    "lgb":      False,
    "catboost": False,
    "gnb":      True,
    "lda":      True,
    "logreg":   True,
    "knn":      True,
    "sgdc":     True,
}

# ─── Logging integration with tqdm ───────────────────────────────────────────────
try:
    from tqdm import tqdm  # noqa
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    # tqdm not installed → use default logger
    pass
