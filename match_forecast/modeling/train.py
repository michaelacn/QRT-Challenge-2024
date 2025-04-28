from pathlib import Path
import os
import yaml
import joblib
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from match_forecast.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, CONFIG_DIR
from match_forecast.modeling.formatters import FORMATTERS

# Map model names to their classes
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
# If TabNet is used
# from pytorch_tabnet.tab_model import TabNetClassifier

MODEL_CLASSES = {
    'rf': RandomForestClassifier,
    'xt': ExtraTreesClassifier,
    'xgb': XGBClassifier,
    'lgb': LGBMClassifier,
    'gnb': GaussianNB,
    'lda': LinearDiscriminantAnalysis,
    'logreg': LogisticRegression,
    'knn': KNeighborsClassifier,
    'catboost': CatBoostClassifier,
    'sgdc': SGDClassifier,
    'tabnet': None  # specify if using TabNet
}

app = typer.Typer()

@app.command()
def main(
    # Paths to features and labels
    features_path: Path = PROCESSED_DATA_DIR / "train_data.csv",
    labels_path: Path = RAW_DATA_DIR / "Y_train.csv",
    # Model selection
    model_name: str = typer.Option(..., help="One of: " + ", ".join(MODEL_CLASSES.keys())),
    # Config & output
    params_path: Path = CONFIG_DIR / "{model_name}_best_params.yaml",
    model_path: Path = MODELS_DIR / "{model_name}_best_model.joblib",
):
    logger.info(f"Training model '{model_name}'...")

    # Ensure directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.debug(f"Loading features from {features_path}")
    X = pd.read_csv(features_path, index_col=0)
    logger.debug(f"Loading labels from {labels_path}")
    y = pd.read_csv(labels_path, index_col=0).squeeze()

    # Load hyperparameters
    params_file = Path(str(params_path).format(model_name=model_name))
    logger.debug(f"Loading hyperparameters from {params_file}")
    with open(params_file, 'r') as f:
        cfg = yaml.safe_load(f)
    raw_params = cfg.get('params', {})

    # Optionally format
    formatter = FORMATTERS.get(model_name)
    if formatter:
        params = formatter(raw_params)
    else:
        params = raw_params

    # Instantiate model
    ModelClass = MODEL_CLASSES.get(model_name)
    if ModelClass is None:
        raise ValueError(f"Unknown or unsupported model: {model_name}")
    model = ModelClass(**params)

    # Train
    for _ in tqdm([None], desc="Fitting model", total=1):
        model.fit(X, y)

    # Save artifact
    model_file = Path(str(model_path).format(model_name=model_name))
    joblib.dump(model, model_file)
    logger.success(f"Model saved to {model_file}")

if __name__ == "__main__":
    app()
