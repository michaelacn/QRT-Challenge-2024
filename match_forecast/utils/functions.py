#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility module for data preprocessing, exploratory analysis, and model evaluation.

Features include:
    - Replacement of string/infinite values with NaN and general null handling.
    - Aggregation and cleaning utilities for team and player statistics.
    - Visualization helpers: missing-value heatmaps, NaN percentage reports.
    - Outlier detection and replacement via Z-score or IQR methods.
    - Imputation strategies for numeric (median) and categorical (mode) data.
    - Quick plotting functions: univariate, bivariate, time series, and frequency charts.
    - Model evaluation tools: cross-validation, performance metrics, and confusion matrices.
    - SHAP-based feature importance and selection.
"""

import numpy as np
import pandas as pd
from typing import List, Literal, Dict, Optional
from tqdm import tqdm

# Machine Learning Imports
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import shap

# Visualization Imports
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics Imports
from scipy import stats
from scipy.stats import pearsonr, spearmanr


# =============================================================================
# Data Cleaning & Aggregation Utilities
# =============================================================================


def replace_null_values(series: pd.Series) -> pd.Series:
    """
    Replace 'None', 'Null', np.inf, and -np.inf with np.nan in a Series.
    """
    return series.replace(["None", "Null", np.inf, -np.inf], np.nan)


def clean_and_impute(
    df: pd.DataFrame, 
    home: bool, 
    meta_cols: Optional[List[str]] = None, 
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Drop metadata, replace nulls, drop sparse cols, then median/categorical-impute
    """
    df = df.drop(columns=meta_cols, errors='ignore').apply(replace_null_values)

    num_cols = df.select_dtypes(include=[np.number]).columns.difference(['ID'])
    sparse = [c for c in num_cols if df[c].isna().sum() > threshold * len(df)]
    df = df.drop(columns=sparse)
    keep_nums = num_cols.difference(sparse)
    if len(keep_nums):
        df[keep_nums] = df[keep_nums].fillna(df[keep_nums].median())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    mode = df[cat_cols].mode()
    if not mode.empty:
        df[cat_cols] = df[cat_cols].fillna(mode.iloc[0])

    prefix = "HOME_" if home else "AWAY_"
    df = df.rename(columns=lambda c: f"{prefix}{c}" if c != "ID" else c)
        
    return df


def agg_positions(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    id_col: str = "ID", 
    pos_col: str = "POSITION"
) -> pd.DataFrame:
    """
    Aggregate numeric features by position group in a single pass.
    """
    df = df.copy()
    df['POSITION_MAP'] = df[pos_col].map(mapping)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference([id_col])
    # Compute mean per (id_col, POSITION_MAP)
    grouped = df.groupby([id_col, 'POSITION_MAP'])[numeric_cols].mean()
    # Pivot to wide format: POSITION_MAP becomes column level
    pivot = grouped.unstack('POSITION_MAP')
    # Flatten MultiIndex columns into single level
    pivot.columns = [f"{feat}_{grp}" for feat, grp in pivot.columns]
    # Reset index to restore `id_col` as a column
    result = pivot.reset_index()
    return result

def merge_and_select_average(
    home_df: pd.DataFrame,
    away_df: pd.DataFrame,
    id_col: str = "ID",
    suffix: str = "_average",
    how: str = "inner"
) -> pd.DataFrame:
    """
    Merge home/away DataFrames on `id_col`, set that as index,
    and keep only columns ending with `suffix`.
    """
    merged = home_df.merge(away_df, on=id_col, how=how)
    merged = merged.set_index(id_col)
    return merged.loc[:, merged.columns.str.endswith(suffix)]


# =============================================================================
# EDA & Outlier Utilities
# =============================================================================


def plot_nan_heatmap(df: pd.DataFrame, title: str) -> None:
    """
    Display a heatmap of missing values.
    """
    plt.figure(figsize=(14, 8))
    sns.heatmap(df.isna(), cbar=False)
    plt.title(title, fontsize=18)
    plt.show()


def print_perc_nans(df: pd.DataFrame, threshold: float) -> None:
    """
    Print column names with a percentage of NaN values above the threshold.
    """
    perc = df.isna().mean() * 100
    filtered = perc[perc > threshold]
    if filtered.empty:
        print("No columns have a percentage of NaN values above the threshold.")
    else:
        for col, p in filtered.items():
            print(f"{col}: {p:.2f}% missing")


def replace_outliers_zscore(series: pd.Series, z_threshold: float = 3.0) -> pd.Series:
    """
    Replace outliers in a numeric Series beyond the z-score threshold with the median.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("The series must be numeric.")

    z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
    median_val = series.median()
    series_clean = series.copy()
    series_clean[z_scores > z_threshold] = median_val
    return series_clean


def replace_outliers_iqr(series: pd.Series) -> pd.Series:
    """
    Replace outliers in a numeric Series using the IQR method with the median.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("The series must be numeric.")

    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    median_val = series.median()
    series_clean = series.copy()
    series_clean[(series < lower) | (series > upper)] = median_val
    return series_clean

# =============================================================================
# Features Engineering
# =============================================================================

def make_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all DIFF_ features by pairing HOME_ and AWAY_ columns.
    """
    home_cols = [c for c in df.columns if c.startswith("HOME_")]
    pairs = [(h, h.replace("HOME_", "AWAY_", 1)) for h in home_cols]
    pairs = [(h, a) for h, a in pairs if a in df.columns]
    for h, a in tqdm(pairs, desc="Diff features"):
        diff_name = h.replace("HOME_", "DIFF_", 1)
        df[diff_name] = df[h] - df[a]
    to_drop = [col for h, a in pairs for col in (h, a)]
    df.drop(columns=to_drop, inplace=True)
    return df

# =============================================================================
# Visualization Functions
# =============================================================================
def plot_frequency_pie_chart(df: pd.DataFrame, col: str, title: str) -> None:
    """
    Display a pie chart of the frequency distribution for a column.
    """
    counts = df[col].value_counts(normalize=True) * 100
    plt.figure(figsize=(8, 6))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "black"},
    )
    plt.title(title, fontsize=16)
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame, title: str, method: Literal["pearson", "kendall", "spearman"] = "pearson"
) -> None:
    """
    Plot the correlation matrix for the DataFrame.
    """
    plt.figure(figsize=(12, 10))
    corr = df.corr(method=method)
    ax = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.xticks(rotation=45, ha="right")
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_univariate_analysis(
    df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]
) -> None:
    """
    Generate univariate plots: histograms for numeric features and bar plots for categorical features.
    """    
    for col in numeric_cols:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title("Histogram")
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title("Boxplot")
        
        plt.tight_layout()
        plt.show()

    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        order = df[col].value_counts().index
        sns.countplot(y=df[col], order=order)
        plt.title("Bar Chart")
        plt.show()


def plot_bivariate_analysis(df: pd.DataFrame) -> None:
    """
    Create simple bivariate scatter plots for numeric features and box plots for numeric vs categorical.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Numeric scatter plots with correlation info
    for i, col1 in enumerate(num_cols):
        for col2 in num_cols[i + 1 :]:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=df[col1], y=df[col2])
            plt.title(f"{col1} vs {col2}")
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.show()

    # Numeric vs Categorical boxplots
    for cat in cat_cols:
        for num in num_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=cat, y=num, data=df)
            plt.title(f"{num} by {cat}")
            plt.show()


def plot_ts(df: pd.DataFrame, time_col: str, cols: List[str]) -> None:
    """
    Plot time series data for the specified columns.
    """
    plt.figure(figsize=(12, 6))
    for col in cols:
        sns.lineplot(x=df[time_col], y=df[col], label=col)
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("Time Series")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_imputed_ts(
    df: pd.DataFrame,
    time_col: str,
    cols: List[str],
    method: Literal["mean", "ffill", "bfill"] = "ffill",
) -> None:
    """
    Plot time series data for specified columns after imputing missing values using the given method.
    """
    df_copy = df.copy()
    plt.figure(figsize=(12, 6))
    for col in cols:
        if method == "mean" and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        else:
            df_copy[col] = df_copy[col].fillna(method=method)
        sns.lineplot(x=df_copy[time_col], y=df_copy[col], label=col)
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("Time Series with Imputation")
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================================================
# Modeling and Evaluation Functions
# =============================================================================
def perform_cross_validation(
    model,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    folds: int,
    scoring: str,
    stratified: bool = False,
) -> None:
    """
    Perform cross-validation using either KFold or StratifiedKFold and print the scores.
    """
    if stratified:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)

    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)

    print(f"Cross-validation {scoring} scores: {scores}")
    print(f"Mean {scoring}: {scores.mean():.4f}, Std: {scores.std():.4f}")


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    title: str,
    model_type: Literal["classifier", "regressor"],
) -> None:
    """
    Evaluate and print model metrics. For classifiers, display the confusion matrix.
    """
    y_pred = model.predict(X_test)

    if model_type == "classifier":
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr")
        gini = 2 * auc_val - 1
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
        plt.title(title, fontsize=16)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
    else:
        gini = 0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Model Performance:")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    if model_type == "classifier":
        print(f"Gini:       {gini:.4f}")


def keep_top_shap_features(X, y, model, shap_folds=3, n_keep=350):
    """
    Computes OOF mean-absolute SHAP importances for each feature and
    retains only the top `n_keep` features by importance.
    """
    skf = StratifiedKFold(n_splits=shap_folds, shuffle=True, random_state=42)
    shap_accum = np.zeros(X.shape[1], dtype=float)
    cols = X.columns

    # 1) OOF SHAP accumulation
    for train_idx, valid_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr = y.iloc[train_idx]

        model.fit(X_tr, y_tr)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_val)

        # multiclass vs binary/reg
        if isinstance(shap_vals, list):
            abs_means = np.mean([np.mean(np.abs(s), axis=0) for s in shap_vals], axis=0)
        else:
            abs_means = np.mean(np.abs(shap_vals), axis=0)

        shap_accum += abs_means

    shap_avg = shap_accum / shap_folds
    shap_imp = pd.Series(shap_avg, index=cols).sort_values(ascending=False)

    # 2) Keep top `n_keep` features
    retained = shap_imp.index[:n_keep].tolist()
    dropped = set(cols) - set(retained)

    return retained, dropped
