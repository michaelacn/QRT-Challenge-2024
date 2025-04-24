from pathlib import Path
import os

import pandas as pd
from loguru import logger
import typer

from match_forecast.config import RAW_DATA_DIR, INTERIM_DATA_DIR
from match_forecast.utils import replace_null_values

app = typer.Typer()

@app.command()
def main(
    # Path to write the team‐level data
    team_output_path: Path = INTERIM_DATA_DIR / "train_data_teams.csv",
    # Path to write the player‐level data
    player_output_path: Path = INTERIM_DATA_DIR / "train_data_players.csv",
    # Path to write the merged dataset
    merged_output_path: Path = INTERIM_DATA_DIR / "train_data_raw.csv",
):
    logger.info("Starting dataset processing...")

    # ─── TEAM STATISTICS ───────────────────────────────────────────────────────────
    logger.info("Processing team statistics…")

    # 1) Ensure output directory exists
    team_output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2) Load raw team CSVs
    df_home = pd.read_csv(RAW_DATA_DIR / "train_home_team_statistics_df.csv", index_col=0)
    df_away = pd.read_csv(RAW_DATA_DIR / "train_away_team_statistics_df.csv", index_col=0)

    # 3) Drop the first two meta columns
    home = df_home.iloc[:, 2:].copy()
    away = df_away.iloc[:, 2:].copy()

    # 4) Tag columns as HOME vs AWAY
    home.columns = [f"HOME_TEAM_{c}" for c in home.columns]
    away.columns = [f"AWAY_TEAM_{c}" for c in away.columns]

    # 5) Concatenate side by side
    df_team = pd.concat([home, away], axis=1, join="inner")

    # 6) Replace nulls with your utility
    df_team = df_team.apply(replace_null_values)

    # 7) Select only "_average" features
    team_processed = df_team.loc[:, df_team.columns.str.endswith("_average")]

    # 8) Write out team dataset
    team_processed.to_csv(team_output_path, index=True)
    logger.success(f"Team data saved to {team_output_path}")

    # ─── PLAYER STATISTICS ─────────────────────────────────────────────────────────
    logger.info("Processing player statistics…")

    # 1) Ensure output directory exists
    player_output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2) Load raw player CSVs
    df_home_p = pd.read_csv(RAW_DATA_DIR / "train_home_player_statistics_df.csv", index_col=0)
    df_away_p = pd.read_csv(RAW_DATA_DIR / "train_away_player_statistics_df.csv", index_col=0)

    # 3) Drop metadata columns (league, team name, player name)
    meta_cols = ['LEAGUE', 'TEAM_NAME', 'PLAYER_NAME']
    df_home_p = df_home_p.drop(columns=meta_cols, errors='ignore')
    df_away_p = df_away_p.drop(columns=meta_cols, errors='ignore')

    # 4) Drop any column with >50% missing values
    frac = 0.5
    min_count = int((1 - frac) * len(df_home_p))
    df_home_p = df_home_p.dropna(axis=1, thresh=min_count)
    min_count = int((1 - frac) * len(df_away_p))
    df_away_p = df_away_p.dropna(axis=1, thresh=min_count)

    # 5) Fill remaining NaNs with column median
    df_home_p = df_home_p.fillna(df_home_p.median())
    df_away_p = df_away_p.fillna(df_away_p.median())

    # 6) Select only "_average" features
    home_avg = df_home_p.loc[:, df_home_p.columns.str.endswith("_average")]
    away_avg = df_away_p.loc[:, df_away_p.columns.str.endswith("_average")]

    # 7) Combine home & away player averages side by side
    player_processed = pd.concat([home_avg, away_avg], axis=1, join="inner")

    # 8) Write out player dataset
    player_processed.to_csv(player_output_path, index=True)
    logger.success(f"Player data saved to {player_output_path}")

    # ─── MERGE TEAM & PLAYER ───────────────────────────────────────────────────────
    logger.info("Merging team and player datasets…")

    # 1) Ensure output directory exists
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2) Merge on the index (match ID or common identifier)
    merged = team_processed.merge(
        player_processed,
        left_index=True,
        right_index=True,
        how='inner'
    )

    # 3) Write out the merged dataset
    merged.to_csv(merged_output_path, index=True)
    logger.success(f"Merged data saved to {merged_output_path}")

if __name__ == "__main__":
    app()
