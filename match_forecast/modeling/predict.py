#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch inference CLI for all trained match_forecast models.

Loads test features, applies each model saved under MODELS_DIR,
and writes out per-model prediction files into MODELS_DIR as
`{model_name}_predictions.csv`.
"""

from pathlib import Path
import joblib
import pandas as pd
import typer
from loguru import logger

from sklearn.model_selection import train_test_split

from match_forecast.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    MODEL_CLASSES,
    TRAIN_SIZE,
    RANDOM_STATE
)

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
    output_dir: Path = typer.Option(
        MODELS_DIR,
        help='Directory to save trained models'
    )
) -> None:
    """
    For each registered model, load its trained artifact,
    run predictions,
    and save the outputs under MODELS_DIR as `<model_name>_predictions.csv`.
    """
    logger.info(f"Loading features from {features_path}")
    X = pd.read_csv(features_path, index_col=0)
    ys = pd.read_csv(labels_path, index_col=0)
    ys = ys.loc[X.index]
    y = ys[['HOME_WINS','DRAW','AWAY_WINS']].idxmax(axis=1)\
          .replace({'HOME_WINS':0, 'DRAW':1, 'AWAY_WINS':2})
    
    # Split into train and (unused) test
    _X_unused, X_test, _y_unused, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Ensure output goes into models_dir
    for model_name in MODEL_CLASSES:
        model_file = output_dir / f"{model_name}_model.joblib"
        if not model_file.exists():
            logger.warning(f"Skipping '{model_name}': model file not found at {model_file}")
            continue

        logger.info(f"Loading model '{model_name}' from {model_file}")
        model = joblib.load(model_file)

        # Base predictions
        logger.info(f"Generating predictions for '{model_name}'...")
        preds = model.predict(X_test)
        df_out = pd.DataFrame(preds, index=X_test.index, columns=["prediction"])
        df_out.index.name = X_test.index.name

        # Write out
        out_file = output_dir / f"{model_name}_predictions.csv"
        df_out.to_csv(out_file, index=True)
        logger.success(f"Saved predictions for '{model_name}' to {out_file}")

if __name__ == "__main__":
    app()
