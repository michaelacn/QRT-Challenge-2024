# QRT-Challenge-2024

### Setup with Python

```bash
python3 -m venv .myenv
source .myenv/bin/activate
pip install -r requirements.txt
```

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