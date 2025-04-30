#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training CLI for match_forecast models with optional preprocessing.

Loads features and labels, applies hyperparameter formatting,
instantiates the specified model, wraps in a scaler+PCA pipeline if needed,
fits it, and saves the artifact.

Usage:
    python -m match_forecast.modeling.train \
        --features-path data/processed/train_data.csv \
        --labels-path data/raw/Y_train.csv \
        --model-name logreg

Outputs:
    MODELS_DIR/{model_name}_best_model.joblib
"""
from pathlib import Path
import yaml
import joblib
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from match_forecast.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    CONFIG_DIR,
    PREPROCESSING_REQUIRED,
    TRAIN_SIZE,
    RANDOM_STATE
)
from match_forecast.modeling.formatters import FORMATTERS

# Model class registry
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier

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
}

app = typer.Typer()

@app.command()
def main(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / 'train_data.csv',
        help='Path to processed feature CSV'
    ),
    labels_path: Path = typer.Option(
        RAW_DATA_DIR / 'Y_train.csv',
        help='Path to training labels CSV'
    ),
    model_name: str = typer.Option(
        ..., help='Model key: ' + ', '.join(MODEL_CLASSES.keys())
    ),
    params_dir: Path = typer.Option(
        CONFIG_DIR, help='Directory containing best_params YAML files'
    ),
    output_dir: Path = typer.Option(
        MODELS_DIR, help='Directory to save trained models'
    )
) -> None:
    """
    Train a model, optionally applying StandardScaler->PCA->StandardScaler
    before fitting if configured in PREPROCESSING_REQUIRED.
    """
    logger.info(f"Training model '{model_name}'...")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    X = pd.read_csv(features_path, index_col=0)

    # Load and encode labels
    ys = pd.read_csv(labels_path, index_col=0)
    ys = ys.loc[X.index]
    onehot = ys[['HOME_WINS', 'DRAW', 'AWAY_WINS']]
    y = onehot.idxmax(axis=1).replace({'HOME_WINS':0,'DRAW':1,'AWAY_WINS':2})

    # Load hyperparameters
    params_file = params_dir / f"{model_name}_best_params.yaml"
    logger.debug(f"Loading params from {params_file}")
    with open(params_file) as f:
        raw_params = yaml.safe_load(f)
    formatter = FORMATTERS.get(model_name)
    params = formatter(raw_params) if formatter else raw_params

    # Instantiate base model
    ModelClass = MODEL_CLASSES.get(model_name)
    if ModelClass is None:
        raise typer.BadParameter(f"Unsupported model: {model_name}")
    params_copy = params.copy()
    n_comp = params_copy.pop('n_components', None)
    base_model = ModelClass(**params_copy)

    # Wrap in preprocessing pipeline if needed
    if PREPROCESSING_REQUIRED.get(model_name, False):
        # default PCA components from params or retain 0.95 variance
        pipeline = Pipeline([
            ('scaler1', StandardScaler()),
            ('pca', PCA(n_components=n_comp, random_state=RANDOM_STATE)),
            ('scaler2', StandardScaler()),
            ('model', base_model)
        ])
        model = pipeline
    else:
        model = base_model

    # Fit model
    logger.info("Fitting model...")
    model.fit(X, y)

    # Save trained model or pipeline
    model_file = output_dir / f"{model_name}_best_model.joblib"
    joblib.dump(model, model_file)
    logger.success(f"Saved trained artifact to {model_file}")

if __name__ == '__main__':
    app()
