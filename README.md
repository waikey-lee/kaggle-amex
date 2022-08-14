## Kaggle Amex Default Prediction Competition
### Intro
- No data lineage tracking / experiment tracking has been implemented from waikey's end haha
- Please follow the steps below to reproduce the features

### Flow
1. Split Raw Data (the big csv gotten from Kaggle) into 5 parquets based on the variable type (2 for delinquency variables)
   * [single notebook](https://github.com/waikey-lee/kaggle-amex/blob/main/notebooks/0a.split_raw_data.ipynb)
   * I store the result in train_parquet and test_parquet which I have [gitignore](https://github.com/waikey-lee/kaggle-amex/blob/main/raw_data/.gitignore) them
2. 