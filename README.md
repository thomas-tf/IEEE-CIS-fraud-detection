https://www.kaggle.com/c/ieee-fraud-detection
# MSBD5001GroupProject
The phases of development as following:
1. Data Analysis
2. Data Cleansing
3. Feature Extraction
    1. One-Hot Encoding
    2. Dimension Reduction (PCA, t-SNE)
4. Models
    1. SVM
    2. CART Decision Tree
    3. LightGBM
    4. XGBoost
5. Model Training
    1. k-fold Cross Validation
    2. Grid Search
6. Evaluation on Benchmarks
    1. Accuracy
    2. Loss
    3. F1-Score
7. Fine-tuning
    1. Model Parameters Tuning

# Get training-ready data
To get feature engineered data, please use the below code (assuming you are working under the /src directory):


```python
from features_generation import feature_engineering

X_train, y_train, X_test, sample_submission = feature_engineering()
```

Note that `feature_engineering()` does not include the following preprocessing:
1. numerical feature scaling
2. fill NaNs

If you model requires such techniques, please implement by yourself.