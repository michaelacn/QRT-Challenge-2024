from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load .env
load_dotenv()

# ─── Project paths ───────────────────────────────────────────────────────────────
PROJ_ROOT          = Path(__file__).resolve().parents[1]
DATA_DIR           = PROJ_ROOT / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
INTERIM_DATA_DIR   = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR  = DATA_DIR / "external"

MODELS_DIR   = PROJ_ROOT / "models"
CONFIG_DIR   = PROJ_ROOT / "config"
REPORTS_DIR  = PROJ_ROOT / "reports"
FIGURES_DIR  = REPORTS_DIR / "figures"

# ─── Split & seed ────────────────────────────────────────────────────────────────
RANDOM_STATE     = 42
TRAIN_SIZE       = 0.8
VALIDATION_SIZE  = 0.2
TEST_SIZE        = 0.2

# ─── Target & Features ───────────────────────────────────────────────────────────
# TARGET_COLUMN     = "AWAY_WINS"
NUMERIC_SUFFIX      = "_average"
CATEGORICAL_COLUMNS = [
    "POSITION"
]

# ─── Player processing settings ──────────────────────────────────────────────────
# Map raw player positions to broader groups
POSITION_MAP = {
    'attacker':   'offensive',
    'midfielder': 'offensive',
    'defender':   'defender',
    'goalkeeper': 'goalkeeper'
}
# Columns to drop when cleaning team stats
META_COLS_TEAMS = ['LEAGUE', 'TEAM_NAME']

# Columns to drop when cleaning player stats
META_COLS_PLAYERS = ['LEAGUE', 'TEAM_NAME', 'PLAYER_NAME']

# ─── Model registry ─────────────────────────────────────────────────────────────
MODEL_NAMES = [
    "rf", "xt", "xgb", "lgb",
    "gnb", "lda", "logreg",
    "knn", "catboost", "sgdc",
    "tabnet"
]

# ─── Logging/TQDM ────────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm  # noqa
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
