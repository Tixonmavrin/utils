import numpy as np
import pandas as pd

from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll.base import scope

import pickle

def _get_result(clf, train_index, test_index, score_func, X, y, group_col, proba, proba_col, use_X, add_params):
    if proba:
      if not proba_col:
        if not use_X:
          if not add_params:
            return score_func(y.iloc[test_index], clf.predict_proba(X.iloc[test_index].drop(columns=[group_col])))
          return score_func(y.iloc[test_index], clf.predict_proba(X.iloc[test_index].drop(columns=[group_col])), add_params)
        else:
          if not add_params:
            return score_func(y.iloc[test_index], clf.predict_proba(X.iloc[test_index].drop(columns=[group_col])), X.iloc[test_index])
          return score_func(y.iloc[test_index], clf.predict_proba(X.iloc[test_index].drop(columns=[group_col])), X.iloc[test_index], add_params)
      else:
        if not use_X:
          if not add_params:
            return score_func(y.iloc[test_index], clf.predict_proba(X.iloc[test_index].drop(columns=[group_col]))[:, proba_col])
          return score_func(y.iloc[test_index], clf.predict_proba(X.iloc[test_index].drop(columns=[group_col]))[:, proba_col], add_params)
        else:
          if not add_params:
            return score_func(y.iloc[test_index], clf.predict_proba(X.iloc[test_index].drop(columns=[group_col]))[:, proba_col], X.iloc[test_index])
          return score_func(y.iloc[test_index], clf.predict_proba(X.iloc[test_index].drop(columns=[group_col]))[:, proba_col], X.iloc[test_index], add_params)
    else:
      if not proba_col:
        if not use_X:
          if not add_params:
            return score_func(y.iloc[test_index], clf.predict(X.iloc[test_index].drop(columns=[group_col])))
          return score_func(y.iloc[test_index], clf.predict(X.iloc[test_index].drop(columns=[group_col])), add_params)
        else:
          if not add_params:
            return score_func(y.iloc[test_index], clf.predict(X.iloc[test_index].drop(columns=[group_col])), X.iloc[test_index])
          return score_func(y.iloc[test_index], clf.predict(X.iloc[test_index].drop(columns=[group_col])), X.iloc[test_index], add_params)
      else:
        if not use_X:
          if not add_params:
            return score_func(y.iloc[test_index], clf.predict(X.iloc[test_index].drop(columns=[group_col]))[:, proba_col])
          return score_func(y.iloc[test_index], clf.predict(X.iloc[test_index].drop(columns=[group_col]))[:, proba_col], add_params)
        else:
          if not add_params:
            return score_func(y.iloc[test_index], clf.predict(X.iloc[test_index].drop(columns=[group_col]))[:, proba_col], X.iloc[test_index])
          return score_func(y.iloc[test_index], clf.predict(X.iloc[test_index].drop(columns=[group_col]))[:, proba_col], X.iloc[test_index], add_params)

def _get_score_stratified(model, fit_params, kf, score_func, X, y, group_col, verbosity, proba, proba_col, use_X, add_params):
  results = []
  for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    clf = model(**fit_params)
    clf.fit(X.iloc[train_index].drop(columns=[group_col]), y.iloc[train_index])
    results.append(_get_result(clf, train_index, test_index, score_func, X, y, group_col, proba, proba_col, use_X, add_params))
  res_mean = np.mean(results)
  return res_mean

def _get_score(model, fit_params, kf, score_func, X, y, group_col, verbosity, proba, proba_col, use_X, add_params):
  results = []
  unique_index = X[group_col].unique()
  for i, (index_train, index_test) in enumerate(kf.split(unique_index, )):
    train_index = X[X[group_col].isin(unique_index[index_train])].index
    test_index = X[~X[group_col].isin(unique_index[index_test])].index

    clf = model(**fit_params)
    clf.fit(X.iloc[train_index].drop(columns=[group_col]), y.iloc[train_index],)
    results.append(_get_result(clf, train_index, test_index, score_func, X, y, group_col, proba, proba_col, use_X, add_params))
  res_mean = np.mean(results)
  return res_mean

def get_best_params(model, search_params, kf, score_func, X, y, num_evals=100, group_col='index', stratified=False, verbosity=0, proba=True, proba_col=None, use_X=False, add_params=False, log_name='logs.txt', trials_file = 'trials.p'):
  '''
  model : sklearn model,
  search_params : dict with hyperopt,
  kf : cross-validation sklearn,
  score_func : loss function,
  X : train features (pandas),
  y : train target (pandas),
  num_evals : num iters,
  group_col : group column for kfold,
  stratified : statified kfold (True or False),
  verbosity : print output (0 or 1),
  proba : return probability of target (True or False),
  use_X : put X to score_func (True or False),
  add_params: put add_params to score_func (None or dict),
  log_name: file for logs (None or string),
  trials_file: file for saving trials (None or string).
  '''
  if log_name:
    try:
      with open(log_name, 'r') as file:
        pass
    except:
      print('No logs')
      with open(log_name, 'w') as file:
        pass
  def hyperopt_score(params):
      try:
        params['cat_features'] = list(params['cat_features'])
        params['add_cat'] = list(params['add_cat'])
      except:
        pass
      try:
        if params['use_cat_new'] == 1:
          params['cat_features'] += params['add_cat']
      except:
        pass
      current_score = _get_score(model, params, kf, score_func, X, y, group_col, verbosity, proba, proba_col, use_X, add_params)
      if log_name:
        with open(log_name, 'a') as file:
          print(current_score, params, file=file)
      else:
        print(current_score, params)
      return {'loss': current_score, 'status': STATUS_OK}
  def hyperopt_stratified_score(params):
      try:
        if params['use_cat_new'] == 1:
          params['cat_features'] += params['add_cat']
      except:
        pass
      current_score = _get_score_stratified(model, params, kf, score_func, X, y, group_col, verbosity, proba, proba_col, use_X, add_params)
      if log_name:
        with open(log_name, 'a') as file:
          print(current_score, params, file=file)
      else:
        print(current_score, params)
      return {'loss': current_score, 'status': STATUS_OK}

  try:
    trials = pickle.load(open(trials_file, "rb"))
  except:
    print('No pickle file')
    trials = Trials()

  if stratified:
    for _ in range(num_evals):
      best = fmin(hyperopt_stratified_score, search_params, tpe.suggest, len(trials)+1, trials, verbose=verbosity, show_progressbar=verbosity)
      pickle.dump(trials, open(trials_file, "wb"))
  else:
    for _ in range(num_evals):
      best = fmin(hyperopt_score, search_params, tpe.suggest, len(trials)+1, trials, verbose=verbosity, show_progressbar=verbosity)
      pickle.dump(trials, open(trials_file, "wb"))
  return best

_search_params_lgb = {
        'class_weight': hp.choice('class_weight', [None, 'balanced']),
        'boosting_type': hp.choice('boosting_type',
                                   ['gbdt','dart','goss']),
        'num_leaves': scope.int(hp.quniform('num_leaves', 20, 200, 1)),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'subsample_for_bin': scope.int(hp.quniform('subsample_for_bin', 20000, 300000, 20000)),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
        'min_data_in_leaf': scope.int(hp.qloguniform('min_data_in_leaf', 0, 6, 1)),
        'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
        'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
        'verbose': -1,
        'subsample': None,
        'reg_alpha': None,
        'reg_lambda': None,
        'min_sum_hessian_in_leaf': None,
        'min_child_samples': None,
        'colsample_bytree': None,
        #'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
        'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
        #'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        #'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        #'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    }
