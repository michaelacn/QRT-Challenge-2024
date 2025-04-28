from pathlib import Path
import os

import pandas as pd
import joblib
from loguru import logger
from tqdm import tqdm
import typer

from match_forecast.config import MODELS_DIR, PROCESSED_DATA_DIR, MODEL_CLASSES

app = typer.Typer()

@app.command()
def main(
    # Path to feature CSV (X_test)
    features_path: Path = PROCESSED_DATA_DIR / "test_data.csv",
    # Name of the model to use for inference
    model_name: str = typer.Option(..., help="Model key, e.g. 'rf', 'xgb', etc."),
    # Path template for writing predictions
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions_{model_name}.csv",
):
    logger.info(f"Loading test features from {features_path}")
    X_test = pd.read_csv(features_path, index_col=0)

    # Ensure predictions directory exists
    predictions_file = Path(str(predictions_path).format(model_name=model_name))
    predictions_file.parent.mkdir(parents=True, exist_ok=True)

    # Load the trained model
    model_file = MODELS_DIR / f"{model_name}_best_model.joblib"
    logger.info(f"Loading model from {model_file}")
    model = joblib.load(model_file)

    # Perform predictions
    logger.info(f"Generating predictions with model '{model_name}'...")
    preds = model.predict(X_test)

    # Build DataFrame for predictions
    df_preds = pd.DataFrame(preds, index=X_test.index, columns=["prediction"])

    # If probability outputs are available, include them
    if hasattr(model, "predict_proba"):
        logger.info("Generating prediction probabilities...")
        proba = model.predict_proba(X_test)
        classes = getattr(model, "classes_", [])
        df_proba = pd.DataFrame(
            proba,
            index=X_test.index,
            columns=[f"prob_{c}" for c in classes]
        )
        df_preds = pd.concat([df_preds, df_proba], axis=1)

    # Save to CSV
    df_preds.to_csv(predictions_file, index=True)
    logger.success(f"Predictions saved to {predictions_file}")

if __name__ == "__main__":
    app()
