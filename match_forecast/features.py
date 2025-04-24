from pathlib import Path
import os

import pandas as pd
from tqdm import tqdm
from loguru import logger
import typer

from match_forecast.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    # Path to the raw merged dataset (intermediate)
    input_path: Path = INTERIM_DATA_DIR / "train_data_raw.csv",
    # Path where the final features dataset will be saved
    output_path: Path = PROCESSED_DATA_DIR / "train_data.csv",
):
    logger.info("Generating features from dataset...")

    # 1) Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2) Load the raw dataset
    df = pd.read_csv(input_path, index_col=0)

    # 3) Compute team-level difference features
    team_home_cols = [c for c in df.columns if c.startswith("HOME_TEAM_")]
    team_pairs = [(h, h.replace("HOME_TEAM_", "AWAY_TEAM_")) for h in team_home_cols]
    # Filter only pairs existing in the DataFrame
    team_pairs = [(h, a) for h, a in team_pairs if a in df.columns]
    for h, a in tqdm(team_pairs, desc="Team diff features"):
        diff_col = h.replace("HOME_TEAM_", "DIFF_TEAM_")
        df[diff_col] = df[h] - df[a]

    # 4) Compute player-level difference features
    player_home_cols = [c for c in df.columns if c.startswith("HOME_PLAYER_")]
    player_pairs = [(h, h.replace("HOME_PLAYER_", "AWAY_PLAYER_")) for h in player_home_cols]
    player_pairs = [(h, a) for h, a in player_pairs if a in df.columns]
    for h, a in tqdm(player_pairs, desc="Player diff features"):
        diff_col = h.replace("HOME_PLAYER_", "DIFF_PLAYER_")
        df[diff_col] = df[h] - df[a]

    # 5) Drop original HOME_* and AWAY_* columns used to compute diffs
    cols_to_drop = [col for pair in team_pairs for col in pair] + \
                   [col for pair in player_pairs for col in pair]
    df.drop(columns=cols_to_drop, inplace=True)

    # 6) Save the features dataset
    df.to_csv(output_path, index=True)
    logger.success(f"Features generation complete. Saved to {output_path}")

if __name__ == "__main__":
    app()
