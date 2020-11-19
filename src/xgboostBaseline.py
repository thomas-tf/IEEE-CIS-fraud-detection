# Reference : https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s

import os
import time
import pdb

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID') # You need to rename the columns

for f in test_identity.columns:
    if f[2] == '-':
        new_name = f[:2] + '_' + f[3:]
        test_identity = test_identity.rename(columns={f: new_name})

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)

y_train = train['isFraud'].copy()
del train_transaction, train_identity, test_transaction, test_identity

# Drop target, fill in NaNs
X_train = train.drop('isFraud', axis=1)
X_test = test.copy()

del train, test

X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))

parameters = {
    'max_depth': [9, 12, 15],
    'learning_rate': [0.02, 0.05, 0.1,],
    'n_estimators': [500, 1000, 2000],
    'subsample': [0.8, 0.85, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'missing': [-999],
    'tree_method': ['gpu_hist']
}

clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    random_state=2019,
    tree_method='gpu_hist'  # THE MAGICAL PARAMETER
)

clf_best_param = xgb.XGBClassifier( # Best score: 0.96864
    n_estimators=500,
    max_depth=15,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.7,
    missing=-999,
    random_state=2019,
    tree_method='gpu_hist'  # THE MAGICAL PARAMETER
)

start_t = time.time()
'''
# gsearch = GridSearchCV(clf, param_grid=parameters, cv=3, n_jobs=4) # it runs several hours
# gsearch.fit(X_train, y_train)

# print("Best score: %0.5f" % gsearch.best_score_)
# print("Best parameters set:")
# best_parameters = gsearch.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))
'''

clf_best_param.fit(X_train, y_train)
sample_submission['isFraud'] = clf_best_param.predict_proba(X_test)[:,1]
sample_submission.to_csv('simple_xgboost.csv')

end_t = time.time()
print('Running time =', end_t - start_t)