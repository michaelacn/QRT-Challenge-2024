#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering CLI for match_forecast project.

Generates difference features (HOME vs AWAY) and selects top features via SHAP:
1) Computes all "DIFF_" features in a single pass and drops original columns.
2) Performs SHAP-based feature selection on training data.
3) Saves selected feature sets for both train and test.

Usage:
    python -m match_forecast.processing.features
    python match_forecast/processing/features.py [OPTIONS]

Outputs:
    PROCESSED_DATA_DIR/train_data.csv
    PROCESSED_DATA_DIR/test_data.csv
"""
from pathlib import Path
import pandas as pd
from loguru import logger
import typer
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

from match_forecast.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_SIZE,
    RANDOM_STATE
)
from match_forecast.utils.functions import make_diff_features, keep_top_shap_features

app = typer.Typer()

@app.command()
def main(
    train_input: Path = INTERIM_DATA_DIR / "train_data_raw.csv",
    y_train_input: Path = RAW_DATA_DIR / "Y_train.csv",
    train_output: Path = PROCESSED_DATA_DIR / "train_data.csv",
    test_input: Path = INTERIM_DATA_DIR / "test_data_raw.csv",
    test_output: Path = PROCESSED_DATA_DIR / "test_data.csv",
    n_keep: int = 275,
):
    """
    Process train & test raw data: diff features + SHAP selection.
    """
    # Ensure output dirs
    for p in (train_output, test_output):
        p.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load and compute diff features
    logger.info("Processing TRAIN diff features")
    df_train = pd.read_csv(train_input, index_col=0)
    df_train = make_diff_features(df_train)
    logger.success("TRAIN diff features completed")

    logger.info("Processing TEST diff features")
    df_test = pd.read_csv(test_input, index_col=0)
    df_test = make_diff_features(df_test)
    logger.success("TEST diff features completed")

    # 2) SHAP-based feature selection on TRAIN
    logger.info("Loading train targets and performing SHAP selection")
    y = pd.read_csv(y_train_input, index_col=0)["AWAY_WINS"].loc[df_train.index]
    X_tr, X_tmp, y_tr, y_tmp = model_selection.train_test_split(
        df_train, y, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )
    rf = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)
    retained, dropped = keep_top_shap_features(X_tr, y_tr, rf, n_keep=n_keep)
    logger.info(f"Retained {len(retained)} features via SHAP, dropped {len(dropped)}")

    # 3) Save selected features
    df_train_sel = df_train[retained]
    df_train_sel.to_csv(train_output)
    df_test_sel = df_test.reindex(columns=retained)
    df_test_sel.to_csv(test_output)
    logger.success(
        f"Selected features saved: TRAIN->{train_output}, TEST->{test_output}"
    )

if __name__ == "__main__":
    app()
