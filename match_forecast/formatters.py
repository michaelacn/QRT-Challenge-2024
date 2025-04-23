import numpy as np

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)  
    elif isinstance(obj, np.floating):
        return float(obj)  
    elif isinstance(obj, list):  
        return [convert_numpy(i) for i in obj]  
    elif isinstance(obj, dict):  
        return {key: convert_numpy(value) for key, value in obj.items()}  
    else:
        return obj

def format_common_int(params, keys):
    for k in keys:
        if k in params and params[k] is not None:
            params[k] = int(params[k])
    return params

def format_rf(params):
    params = format_common_int(params, ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes'])
    if 'bootstrap' in params:
        params['bootstrap'] = bool(params['bootstrap'])
    class_weights = [None, 'balanced', 'balanced_subsample']
    criterion_options = ['gini', 'entropy']
    if 'class_weight' in params:
        params['class_weight'] = class_weights[int(params['class_weight'])] if isinstance(params['class_weight'], (int, np.integer)) else params['class_weight']
    if 'criterion' in params:
        params['criterion'] = criterion_options[int(params['criterion'])] if isinstance(params['criterion'], (int, np.integer)) else params['criterion']
    return params

def format_xt(params):
    return format_rf(params)

def format_xgb(params):
    params = format_common_int(params, ['max_depth', 'n_estimators', 'min_child_weight'])
    return params

def format_lgb(params):
    params = format_common_int(params, ['max_depth', 'num_leaves', 'n_estimators', 'min_child_samples'])
    return params

def format_catboost(params):
    params = format_common_int(params, ['iterations', 'depth', 'border_count'])
    grow_policy_options = ['SymmetricTree', 'Depthwise', 'Lossguide']
    auto_weights_options = [None, 'Balanced']
    if 'grow_policy' in params:
        params['grow_policy'] = grow_policy_options[int(params['grow_policy'])] if isinstance(params['grow_policy'], (int, np.integer)) else params['grow_policy']
    if 'auto_class_weights' in params:
        params['auto_class_weights'] = auto_weights_options[int(params['auto_class_weights'])] if isinstance(params['auto_class_weights'], (int, np.integer)) else params['auto_class_weights']
    return params

def format_logreg(params):
    params = format_common_int(params, ['n_components'])
    return params

def format_sgdc(params):
    params = format_common_int(params, ['n_components'])
    return params

def format_qda(params):
    params = format_common_int(params, ['n_components'])
    return params

def format_gnb(params, y_train=None):
    params = format_common_int(params, ['n_components'])
    learn_priors = bool(params.get('learn_priors', False))
    if learn_priors and y_train is not None:
        n_classes = len(np.unique(y_train))
        params['priors'] = [1.0 / n_classes] * n_classes
    else:
        params['priors'] = None
    return params

def format_knn(params):
    weights_list = ['uniform', 'distance']
    algorithm_list = ['auto', 'ball_tree', 'kd_tree', 'brute']
    params = format_common_int(params, ['n_neighbors', 'leaf_size', 'n_components'])
    if 'weights' in params:
        params['weights'] = weights_list[int(params['weights'])] if isinstance(params['weights'], (int, np.integer)) else params['weights']
    if 'algorithm' in params:
        params['algorithm'] = algorithm_list[int(params['algorithm'])] if isinstance(params['algorithm'], (int, np.integer)) else params['algorithm']
    return params

def format_lda(params):
    solver_list = ['lsqr', 'eigen']
    params = format_common_int(params, ['n_components'])
    if 'solver' in params:
        params['solver'] = solver_list[int(params['solver'])] if isinstance(params['solver'], (int, np.integer)) else params['solver']
    if 'shrinkage_val' in params and params['shrinkage_val'] is not None:
        params['shrinkage'] = float(params['shrinkage_val'])
    return params

def format_tabnet(params):
    return format_common_int(
        params,
        ['n_d', 'n_a', 'n_steps', 'n_shared', 'n_independent']
    )

FORMATTERS = {
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
    'qda': format_qda
}
