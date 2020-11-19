# Reference : https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s

import xgboost as xgb
from features_generation import feature_engineering

X_train, y_train, X_test, sample_submission = feature_engineering()

X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

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

clf.fit(X_train, y_train)
sample_submission['isFraud'] = clf.predict_proba(X_test)[:, 1]
sample_submission.to_csv('simple_xgboost.csv')
