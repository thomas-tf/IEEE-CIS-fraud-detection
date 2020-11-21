# Reference : https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s

import time
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from features_generation import feature_engineering

X_train, y_train, X_test, sample_submission = feature_engineering()

X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

parameters = {
    'max_depth': [9, 12, 15],
    'learning_rate': [0.02, 0.05, 0.1],
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

# clf_best_param = xgb.XGBClassifier( # Best score: 0.96864
#     n_estimators=500,
#     max_depth=15,
#     learning_rate=0.02,
#     subsample=0.85,
#     colsample_bytree=0.7,
#     missing=-999,
#     random_state=2019,
#     tree_method='gpu_hist'  # THE MAGICAL PARAMETER
# )

start_t = time.time()

gsearch = GridSearchCV(clf, param_grid=parameters, cv=3, n_jobs=4)  # it runs several hours
gsearch.fit(X_train, y_train)
print("Best score: %0.5f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# clf_best_param.fit(X_train, y_train)
# sample_submission['isFraud'] = clf_best_param.predict_proba(X_test)[:,1]
# sample_submission.to_csv('simple_xgboost.csv')

end_t = time.time()
print('Running time =', end_t - start_t)
