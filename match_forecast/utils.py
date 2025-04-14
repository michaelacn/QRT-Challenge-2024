#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified Data Preprocessing, Visualization, and Modeling Module

This module includes simplified functions to:
    - Replace string representations like "None" and "Null" with np.nan.
    - Visualize missing values.
    - Identify columns with high percentages of missing data.
    - Replace outlier values using Z-score or IQR methods.
    - Impute missing values using different strategies.
    - Create quick univariate and bivariate plots.
    - Plot time series data (with optional missing value handling).
    - Perform cross-validation and evaluate model performance.
"""

import numpy as np
import pandas as pd
from typing import List, Literal

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
# Data Preprocessing and Utility Functions
# =============================================================================


def replace_null_values(series: pd.Series) -> pd.Series:
    """
    Replace 'None' and 'Null' with np.nan in a Series.
    """
    return series.replace(["None", "Null"], np.nan)


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
    for col, p in perc[perc > threshold].items():
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


def impute_missing_values(
    series: pd.Series, method: Literal["mean", "median", "ffill", "bfill"]
) -> pd.Series:
    """
    Impute missing values in a Series using the chosen method.
    """
    if method == "mean":
        if pd.api.types.is_numeric_dtype(series):
            return series.fillna(series.mean())
        else:
            raise ValueError("Mean imputation is applicable only for numeric series.")
    elif method == "median":
        if pd.api.types.is_numeric_dtype(series):
            return series.fillna(series.median())
        else:
            raise ValueError("Median imputation is applicable only for numeric series.")
    elif method in ["ffill", "bfill"]:
        return series.fillna(method=method)
    else:
        raise ValueError("Invalid imputation method. Use 'mean', 'median', 'ffill' or 'bfill'.")


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
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(method=method), annot=True, cmap="coolwarm", fmt=".2f")
    plt.xticks(rotation=45)
    plt.title(title, fontsize=18)
    plt.show()


def plot_univariate_analysis(
    df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]
) -> None:
    """
    Generate univariate plots: histograms for numeric features and bar plots for categorical features.
    """
    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram of {col}")
        plt.show()

    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        order = df[col].value_counts().index
        sns.countplot(y=df[col], order=order)
        plt.title(f"Bar Chart of {col}")
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
