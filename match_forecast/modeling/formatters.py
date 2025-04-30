#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter formatting utilities for match_forecast project.

This module provides:
  - `convert_numpy`: convert numpy types to native Python types.
  - `format_common_int`: helper to cast specified keys to int.
  - `format_<model>`: functions to transform raw hyperopt parameter dicts
    into valid sklearn/CatBoost constructor arguments.
  - `FORMATTERS` registry mapping model names to formatter functions.
"""

import numpy as np
from typing import Dict, Any, Optional


def convert_numpy(obj: Any) -> Any:
    """
    Recursively convert numpy scalar types in lists/dicts to native Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    return obj


def format_common_int(params: Dict[str, Any], keys: list[str]) -> Dict[str, Any]:
    """
    Cast specified keys in params to int if present and not None.
    """
    for key in keys:
        if key in params and params[key] is not None:
            params[key] = int(params[key])
    return params


def format_rf(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format RandomForest parameters, cast ints, map options, inject defaults.
    """
    params = format_common_int(
        params,
        ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes']
    )
    # Map booleans
    if 'bootstrap' in params:
        params['bootstrap'] = bool(params['bootstrap'])
    # Map indices
    class_weights = [None, 'balanced', 'balanced_subsample']
    if 'class_weight' in params and isinstance(params['class_weight'], (int, np.integer)):
        params['class_weight'] = class_weights[int(params['class_weight'])]
    criterion_opts = ['gini', 'entropy']
    if 'criterion' in params and isinstance(params['criterion'], (int, np.integer)):
        params['criterion'] = criterion_opts[int(params['criterion'])]
    # Inject defaults
    defaults = {'n_jobs': -1, 'random_state': 42}
    for k, v in defaults.items():
        params.setdefault(k, v)
    return params


def format_xt(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Alias to format_rf for ExtraTreesClassifier.
    """
    return format_rf(params)


def format_xgb(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format XGBoost parameters, cast ints, and inject multiclass defaults.
    """
    params = format_common_int(params, ['max_depth', 'n_estimators', 'min_child_weight'])
    # Inject defaults for XGB multiclass
    defaults = {'objective': 'multi:softmax', 'num_class': 3, 'n_jobs': -1}
    for k, v in defaults.items():
        params.setdefault(k, v)
    return params


def format_lgb(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format LightGBM parameters, cast ints, inject multiclass defaults.
    """
    params = format_common_int(params, ['max_depth', 'num_leaves', 'n_estimators', 'min_child_samples'])
    defaults = {
        'objective': 'multiclass',
        'boosting_type': 'gbdt',
        'num_class': 3,
        'n_jobs': -1,
        'verbose': -1
    }
    for k, v in defaults.items():
        params.setdefault(k, v)
    return params


def format_catboost(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format CatBoost parameters, cast ints, map indices, inject multiclass defaults.
    """
    params = format_common_int(params, ['iterations', 'depth', 'border_count'])
    # Map enums
    grow_opts = ['SymmetricTree', 'Depthwise', 'Lossguide']
    if 'grow_policy' in params and isinstance(params['grow_policy'], (int, np.integer)):
        params['grow_policy'] = grow_opts[int(params['grow_policy'])]
    auto_opts = [None, 'Balanced']
    if 'auto_class_weights' in params and isinstance(params['auto_class_weights'], (int, np.integer)):
        params['auto_class_weights'] = auto_opts[int(params['auto_class_weights'])]
    # Inject defaults
    defaults = {
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'verbose': False,
        'thread_count': -1,
        'random_seed': 42
    }
    for k, v in defaults.items():
        params.setdefault(k, v)
    return params


def format_logreg(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format LogisticRegression params, cast ints, inject defaults for saga elasticnet.
    """
    params = format_common_int(params, ['n_components'])

    # Keep C & l1_ratio from params
    defaults = {
        'penalty': 'elasticnet',
        'solver': 'saga',
        'max_iter': 2000,
        'random_state': 42
    }
    for k, v in defaults.items():
        params.setdefault(k, v)
    return params


def format_sgdc(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format SGDClassifier params, inject modified_huber elasticnet defaults.
    """
    params = format_common_int(params, ['n_components'])

    defaults = {
        'loss': 'modified_huber',
        'penalty': 'elasticnet',
        'max_iter': 1000,
        'tol': 1e-3,
        'n_jobs': -1,
        'random_state': 42
    }
    for k, v in defaults.items():
        params.setdefault(k, v)
    return params


def format_qda(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format QDA parameters (e.g., n_components).
    """
    params = format_common_int(params, ['n_components'])
    return params


def format_gnb(params: Dict[str, Any], y_train: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Format GaussianNB parameters, optionally computing priors if requested.
    """
    params = format_common_int(params, ['n_components'])
    
    # No n_components but handle learn_priors flag
    if 'learn_priors' in params:
        learn = bool(params.pop('learn_priors'))
        if learn and y_train is not None:
            n_classes = len(np.unique(y_train))
            params['priors'] = [1.0 / n_classes] * n_classes
        else:
            params['priors'] = None
    return params


def format_knn(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format KNeighborsClassifier params, cast ints, map indices, inject defaults.
    """
    params = format_common_int(params, ['n_neighbors', 'leaf_size', 'p'])
    # Map enums
    weights_opts = ['uniform', 'distance']
    if 'weights' in params and isinstance(params['weights'], (int, np.integer)):
        params['weights'] = weights_opts[int(params['weights'])]
    alg_opts = ['auto', 'ball_tree', 'kd_tree', 'brute']
    if 'algorithm' in params and isinstance(params['algorithm'], (int, np.integer)):
        params['algorithm'] = alg_opts[int(params['algorithm'])]
    # Inject defaults
    params.setdefault('n_jobs', -1)
    return params


def format_lda(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format LinearDiscriminantAnalysis parameters: cast n_components and solver/shrinkage.
    """
    params = format_common_int(params, ['n_components'])

    solver_opts = ['lsqr', 'eigen']
    if 'solver' in params and isinstance(params['solver'], (int, np.integer)):
        params['solver'] = solver_opts[int(params['solver'])]

    if 'shrinkage_val' in params and params['shrinkage_val'] is not None:
        params['shrinkage'] = float(params.pop('shrinkage_val'))

    return params


def format_tabnet(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format TabNet parameters by casting key ints.
    """
    return format_common_int(
        params,
        ['n_d', 'n_a', 'n_steps', 'n_shared', 'n_independent']
    )

# Registry of all formatters
FORMATTERS: Dict[str, Any] = {
    'rf': format_rf,
    'xt': format_xt,
    'xgb': format_xgb,
    'lgb': format_lgb,
    'gnb': format_gnb,
    'lda': format_lda,
    'logreg': format_logreg,
    'knn': format_knn,
    'catboost': format_catboost,
    'sgdc': format_sgdc,
    'qda': format_qda,
    'tabnet': format_tabnet
}
