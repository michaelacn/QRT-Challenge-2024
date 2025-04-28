#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset preprocessing CLI script for match_forecast project.

This script generates intermediate CSVs for both training and testing sets:

- Team-level data: cleans, imputes, and prefixes home/away team statistics, then merges and selects `_average` features.
- Player-level data: cleans, imputes, aggregates player statistics by position groups (offensive, defender, goalkeeper), prefixes, and merges home/away.
- Merged raw datasets: combines team and player processed data into a single dataset for training and testing.

Outputs:
    INTERIM_DATA_DIR/train_data_teams.csv
    INTERIM_DATA_DIR/train_data_players.csv
    INTERIM_DATA_DIR/train_data_raw.csv
    INTERIM_DATA_DIR/test_data_teams.csv
    INTERIM_DATA_DIR/test_data_players.csv
    INTERIM_DATA_DIR/test_data_raw.csv

Usage:
    python -m match_forecast.processing.dataset
    (or)
    python match_forecast/processing/dataset.py [OPTIONS]

Options can override default paths for any of the six output files.
"""

from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from match_forecast.config import *
from match_forecast.utils.functions import *

app = typer.Typer()

@app.command()
def main(
    # TRAIN paths
    train_team_output: Path = INTERIM_DATA_DIR / "train_data_teams.csv",
    train_player_output: Path = INTERIM_DATA_DIR / "train_data_players.csv",
    train_merged_output: Path = INTERIM_DATA_DIR / "train_data_raw.csv",
    # TEST paths
    test_team_output: Path = INTERIM_DATA_DIR / "test_data_teams.csv",
    test_player_output: Path = INTERIM_DATA_DIR / "test_data_players.csv",
    test_merged_output: Path = INTERIM_DATA_DIR / "test_data_raw.csv",
):
    logger.info("Starting dataset processing for TRAIN set…")
    # Ensure output dirs
    for p in [train_team_output, train_player_output, train_merged_output,
              test_team_output, test_player_output, test_merged_output]:
        p.parent.mkdir(parents=True, exist_ok=True)

    # --- TRAIN TEAM ---
    logger.info("Processing TRAIN team statistics…")
    df_home = pd.read_csv(RAW_DATA_DIR / "train_home_team_statistics_df.csv", index_col=0)
    df_away = pd.read_csv(RAW_DATA_DIR / "train_away_team_statistics_df.csv", index_col=0)

    meta_cols_teams = ['LEAGUE', 'TEAM_NAME']
    home_teams = clean_and_impute(df_home, home=True, meta_cols=meta_cols_teams)
    away_teams = clean_and_impute(df_away, home=False, meta_cols=meta_cols_teams)

    train_data_teams = merge_and_select_average(home_teams, away_teams)
    train_data_teams.to_csv(train_team_output, index=True)
    logger.success(f"Train team data saved to {train_team_output}")

   # --- TRAIN PLAYER ---
    logger.info("Processing TRAIN player statistics…")
    df_home_p = pd.read_csv(RAW_DATA_DIR / "train_home_player_statistics_df.csv", index_col=0)
    df_away_p = pd.read_csv(RAW_DATA_DIR / "train_away_player_statistics_df.csv", index_col=0)

    home_players = clean_and_impute(df_home_p, home=True, meta_cols=META_COLS_PLAYERS)
    home_agg = agg_positions(home_players, mapping=POSITION_MAP)

    away_players = clean_and_impute(df_away_p, home=False, meta_cols=META_COLS_PLAYERS)
    away_agg = agg_positions(away_players, mapping=POSITION_MAP)

    train_data_players = merge_and_select_average(home_agg, away_agg)
    train_data_players.to_csv(train_player_output, index=True)
    logger.success(f"Train player data saved to {train_player_output}")

    # --- FULL TRAIN DATASET ---
    logger.info("Merging TRAIN team & player datasets…")
    train_merged = train_data_teams.merge(train_data_players, left_index=True, right_index=True, how='inner')
    train_merged.to_csv(train_merged_output, index=True)
    logger.success(f"Train merged data saved to {train_merged_output}")

    # --- TEST TEAM ---
    logger.info("Processing TEST team statistics…")
    df_home_t = pd.read_csv(RAW_DATA_DIR / "test_home_team_statistics_df.csv", index_col=0)
    df_away_t = pd.read_csv(RAW_DATA_DIR / "test_away_team_statistics_df.csv", index_col=0)

    home_teams_t = clean_and_impute(df_home_t, home=True, meta_cols=None)
    away_teams_t = clean_and_impute(df_away_t, home=False, meta_cols=None)

    test_data_teams = merge_and_select_average(home_teams_t, away_teams_t)
    test_data_teams.to_csv(test_team_output, index=True)
    logger.success(f"Test team data saved to {test_team_output}")

    # --- TEST PLAYER ---
    logger.info("Processing TEST player statistics…")
    df_home_p_t = pd.read_csv(RAW_DATA_DIR / "test_home_player_statistics_df.csv", index_col=0)
    df_away_p_t = pd.read_csv(RAW_DATA_DIR / "test_away_player_statistics_df.csv", index_col=0)

    home_players_t = clean_and_impute(df_home_p_t, home=True, meta_cols=None)
    home_agg_t = agg_positions(home_players_t, mapping=POSITION_MAP)

    away_players_t = clean_and_impute(df_away_p_t, home=False, meta_cols=None)
    away_agg_t = agg_positions(away_players_t, mapping=POSITION_MAP)

    test_data_players = merge_and_select_average(home_agg_t, away_agg_t)
    test_data_players.to_csv(test_player_output, index=True)
    logger.success(f"Test player data saved to {test_player_output}")

    # --- TEST MERGED ---
    logger.info("Merging TEST team & player datasets…")
    test_merged = test_data_teams.merge(test_data_players, left_index=True, right_index=True, how='inner')
    test_merged.to_csv(test_merged_output, index=True)
    logger.success(f"Test merged data saved to {test_merged_output}")

if __name__ == "__main__":
    app()
