# QRT-Challenge-2024

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Challenge match forecast

## Project Organization

```
├── LICENSE
├── Makefile                  <- Utility commands (e.g., `make data`, `make train`)
├── README.md                 
├── data
│   ├── external              <- Third-party/raw source data                
│   ├── interim               <- Transformed intermediate datasets
│   ├── processed             <- Final cleaned datasets for modeling
│   └── raw                   <- Original datasets
│
├── docs                      <- Static site documentation (e.g., MkDocs)
├── models                    <- Trained model artifacts
├── notebooks                 <- Jupyter notebooks for exploration and analysis
├── pyproject.toml            <- Package metadata and tool configuration
├── requirements.txt          
└── match_forecast
    ├── __init__.py
    ├── config.py             <- Global constants and path definitions
    │
    ├── processing
    │   ├── __init__.py
    │   ├── dataset.py        <- CLI to generate and clean intermediate datasets
    │   └── features.py       <- CLI and utilities for feature engineering
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── formatters.py     <- Hyperparameter formatting per algorithm
    │   ├── predict.py        <- Inference script using saved models
    │   └── train.py          <- CLI to train models (with optional scaling/PCA)
    │
    └── utils
        ├── __init__.py
        ├── functions.py      <- Data cleaning, outlier handling, SHAP selection, etc.
        └── plots.py          <- Reusable plotting functions

```

--------