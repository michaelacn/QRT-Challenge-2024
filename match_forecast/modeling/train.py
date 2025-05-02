#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training CLI for match_forecast models with optional preprocessing.

Loads features and labels once, splits into train/test, then for each specified model (or all models),
applies hyperparameter formatting, instantiates the model (with optional
StandardScaler->PCA->StandardScaler pipeline), fits it on the training split,
and saves the artifact.

Usage:
    # Single model
    python -m match_forecast.modeling.train \
        --features-path data/processed/train_data.csv \
        --labels-path data/raw/Y_train.csv \
        --model-name rf

    # Multiple models or all
    python -m match_forecast.modeling.train \
        --features-path data/processed/train_data.csv \
        --labels-path data/raw/Y_train.csv \
        --model-name rf --model-name xgb --model-name lgb

    # Or simply train all:
    python -m match_forecast.modeling.train --all

Outputs:
    MODELS_DIR/{model_name}_model.joblib
"""

from pathlib import Path
import yaml
import joblib
import pandas as pd
from loguru import logger
import typer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from match_forecast.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    CONFIG_DIR,
    PREPROCESSING_REQUIRED,
    RANDOM_STATE,
    TRAIN_SIZE
)
from match_forecast.modeling.formatters import FORMATTERS

# Model class registry
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
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
    'qda': QuadraticDiscriminantAnalysis,
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
    model_name: list[str] = typer.Option(
        None,
        "--model-name", "-m",
        help='Model key(s). Repeatable. Ignored if --all is set.'
    ),
    all_models: bool = typer.Option(
        False,
        "--all",
        help='If set, trains all supported models.'
    ),
    params_dir: Path = typer.Option(
        CONFIG_DIR,
        help='Directory containing best_params YAML files'
    ),
    output_dir: Path = typer.Option(
        MODELS_DIR,
        help='Directory to save trained models'
    )
) -> None:
    """
    Train one or more models, optionally applying a preprocessing pipeline
    before fitting if required by the model, on an 80% train split.
    """
    # Determine which models to train
    if all_models:
        names = list(MODEL_CLASSES.keys())
    elif model_name:
        names = model_name
    else:
        raise typer.BadParameter("Provide --model-name or --all.")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    X = pd.read_csv(features_path, index_col=0)
    ys = pd.read_csv(labels_path, index_col=0)
    ys = ys.loc[X.index]
    y = ys[['HOME_WINS','DRAW','AWAY_WINS']].idxmax(axis=1)\
          .replace({'HOME_WINS':0, 'DRAW':1, 'AWAY_WINS':2})

    # Split into train and (unused) test
    X_train, _X_unused, y_train, _y_unused = train_test_split(
        X, y, train_size=TRAIN_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    for name in names:
        logger.info(f"Training '{name}'...")
        # Load and format hyperparameters
        pfile = params_dir / f"{name}_params.yaml"
        logger.debug(f"Loading params from {pfile}")
        with open(pfile) as f:
            raw = yaml.safe_load(f)
        formatter = FORMATTERS.get(name)
        params = formatter(raw) if formatter else raw

        # Instantiate base model
        ModelClass = MODEL_CLASSES.get(name)
        if ModelClass is None:
            logger.error(f"Unsupported model '{name}'")
            continue
        pcopy = params.copy()
        n_comp = pcopy.pop('n_components', None)
        base = ModelClass(**pcopy)

        # Build pipeline if needed
        if PREPROCESSING_REQUIRED.get(name, False):
            model = Pipeline([
                ('scaler1', StandardScaler()),
                ('pca', PCA(n_components=n_comp, random_state=RANDOM_STATE)),
                ('scaler2', StandardScaler()),
                ('model', base)
            ])
        else:
            model = base

        # Fit and save
        logger.info("Fitting model...")
        model.fit(X_train, y_train)
        outfile = output_dir / f"{name}_model.joblib"
        joblib.dump(model, outfile)
        logger.success(f"Saved {name} to {outfile}")

if __name__ == '__main__':
    app()
