https://www.kaggle.com/c/ieee-fraud-detection
# MSBD5001GroupProject
The phases of development as following:
1. Data Analysis
2. Data Cleansing
3. Feature Extraction
3.1 One-Hot Encoding
3.2 Dimension Reduction (PCA, t-SNE)
4. Models
4.1. SVM
4.2. CART Decision Tree
4.3. LightGBM
4.4. XGBoost
5. Model Training
5.1. k-fold Cross Validation
5.2. Grid Search
6. Evaluation on Benchmarks
6.1. Accuracy
6.2. Loss
6.3. F1-Score
7. Fine-tuning
7.1. Model Parameters Tuning

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