#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training CLI for match_forecast models with optional preprocessing.

Loads features and labels once, splits into train/test, then for each model,
applies hyperparameter formatting, instantiates the model (with optional
StandardScaler->PCA->StandardScaler pipeline), fits it on the training split,
and saves the artifact.

Usage:
    python -m match_forecast.modeling.train

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
    MODEL_CLASSES,
    PREPROCESSING_REQUIRED,
    RANDOM_STATE,
    TRAIN_SIZE
)
from match_forecast.modeling.formatters import FORMATTERS

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
    Train models, optionally applying a preprocessing pipeline
    before fitting if required by the model.
    """
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

    for model_name in MODEL_CLASSES:
        logger.info(f"Training '{model_name}'...")
        # Load and format hyperparameters
        pfile = params_dir / f"{model_name}_params.yaml"
        logger.debug(f"Loading params from {pfile}")
        with open(pfile) as f:
            raw = yaml.safe_load(f)
        formatter = FORMATTERS.get(model_name)
        params = formatter(raw) if formatter else raw

        # Instantiate base model
        ModelClass = MODEL_CLASSES.get(model_name)
        if ModelClass is None:
            logger.error(f"Unsupported model '{model_name}'")
            continue
        pcopy = params.copy()
        n_comp = pcopy.pop('n_components', None)
        base = ModelClass(**pcopy)

        # Build pipeline if needed
        if PREPROCESSING_REQUIRED.get(model_name, False):
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
        outfile = output_dir / f"{model_name}_model.joblib"
        joblib.dump(model, outfile)
        logger.success(f"Saved {model_name} to {outfile}")

if __name__ == '__main__':
    app()
