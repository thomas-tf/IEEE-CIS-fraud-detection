https://www.kaggle.com/c/ieee-fraud-detection
# MSBD5001GroupProject
The phases of development as following:
1. Data Analysis
2. Data Cleansing
3. Feature Extraction <br/>
&nbsp; &nbsp; &nbsp; &nbsp; 3.1 One-Hot Encoding<br/>
&nbsp; &nbsp; &nbsp; &nbsp; 3.2 Dimension Reduction (PCA, t-SNE)
4. Models<br/>
&nbsp; &nbsp; &nbsp; &nbsp; 4.1. SVM<br/>
&nbsp; &nbsp; &nbsp; &nbsp; 4.2. CART Decision Tree<br/>
&nbsp; &nbsp; &nbsp; &nbsp; 4.3. LightGBM<br/>
&nbsp; &nbsp; &nbsp; &nbsp; 4.4. XGBoost
5. Model Training<br/>
&nbsp; &nbsp; &nbsp; &nbsp; 5.1. k-fold Cross Validation<br/>
&nbsp; &nbsp; &nbsp; &nbsp; 5.2. Grid Search
6. Evaluation on Benchmarks<br/>
&nbsp; &nbsp; &nbsp; &nbsp; 6.1. Accuracy<br/>
&nbsp; &nbsp; &nbsp; &nbsp; 6.2. Loss<br/>
&nbsp; &nbsp; &nbsp; &nbsp; 6.3. F1-Score
7. Fine-tuning<br/>
&nbsp; &nbsp; &nbsp; &nbsp; 7.1. Model Parameters Tuning


### Quick Start

```bash
DOCKER_BUILDKIT=1 docker build -t lab .
docker run --rm -it -v $PWD:/workdir -w /workdir -p 20080:8888 lab bash
jupyter-lab --ip='*' --NotebookApp.token='' --NotebookApp.password='' --allow-root
```