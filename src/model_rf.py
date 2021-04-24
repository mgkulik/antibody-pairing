#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 07:23:00 2021

@author: magoncal
"""

import pandas as pd
import numpy as np
import random

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import metrics

data_embeddings = pd.read_csv('embeddings_first_1000_Antibodies.csv').reset_index()
data_embeddings['index'] = "paired_"+data_embeddings['index'].astype(str)
data_embeddings = data_embeddings.drop(columns=['id'])
data_embeddings['class'] = 1

seed = 12345

def randomize_rows(data_embeddings, seed):
    random.seed(seed)
    row_ids = random.sample(range(0, len(data_embeddings)), len(data_embeddings))
    
    cols = list(data_embeddings.columns)[1:-400]
    data_paired = data_embeddings.loc[:, cols]
    data_paired['id'] = row_ids
    
    cols = list(data_embeddings.columns)[-400:]
    data_schuffled = data_embeddings.loc[row_ids, cols]
    data_schuffled['id'] = row_ids
    data_schuffled['index'] = "shuffled_"+data_schuffled['id'].astype(str)
    
    new_embeddings = pd.merge(data_paired, data_schuffled, on='id')
    new_embeddings = new_embeddings.drop(columns=['id'])
    new_embeddings['class'] = 0
    
    data_all_embeddings = pd.concat([data_embeddings, new_embeddings])
    return data_all_embeddings


def get_best_model(algo, params, X_train, y_train, st=0, verb=1, cv=5, iters=100):
    ''' Function to apply Tunning Search for the estimator parameters. To the slow ones I will try using RandomSearchCV. 
    Otherwise I will use GridSearchCV.
    Parms:
    algo - Estimator instance
    params - Parameters to try as a dict with name of argument as key and list of values to test
    X_train - Train Data
    y_train - Train classes
    st - Type of Search - 0: GridSearchCV (default) , 1: RandomizedSearchCV
    verb - 1 to print iterations results and 0 to silent execution
    cv - Number of Cross-Validation groups
    iters - Max iterations for RandomSearchCV
    '''
    if (st==0):
        mdl_search = GridSearchCV(estimator=algo, param_grid = params, 
                                  scoring='roc_auc', cv=cv, return_train_score=True, n_jobs=-1, verbose=verb)
    else:
        mdl_search = RandomizedSearchCV(estimator=algo, param_distributions = params, 
                                    scoring='roc_auc', cv=cv, return_train_score=True, n_jobs=-1, verbose=verb, n_iter=iters)
    
    mdl_search.fit(X_train, y_train.ravel())
    print('Best Params: {}'.format(mdl_search.best_params_))
    return mdl_search.best_estimator_


data_all_embeddings = randomize_rows(data_embeddings, seed)

cols=list(data_all_embeddings.columns)[1:-1]
X=data_all_embeddings.loc[:, cols]  # Features
y=data_all_embeddings['class']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

rf_params = {'bootstrap': [True, False],
             'max_depth': [10, 30, 60, 90, 100, None],
             'max_features': ['auto', 'sqrt'],
             'min_samples_leaf': [1, 2, 4],
             'min_samples_split': [2, 5, 10],
             'n_estimators': [200, 400]}

rf_params = {'bootstrap': [True, False],
             'max_depth': [10, 90, None],
             'max_features': ['auto', 'sqrt'],
             'min_samples_leaf': [1, 4],
             'min_samples_split': [2, 10],
             'n_estimators': [100, 200]}

rf = RandomForestClassifier()
best_rf_model = get_best_model(rf, rf_params, X_train, y_train)

best_rf_model.fit(X_train,y_train)

y_check=best_rf_model.predict(X_train)
y_pred=best_rf_model.predict(X_test)

print("Accuracy over the Train Set:", metrics.accuracy_score(y_train, y_check))
print("Accuracy over the Test Set:", metrics.accuracy_score(y_test, y_pred))
