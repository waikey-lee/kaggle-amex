## Kaggle Amex Default Prediction Competition
### Intro
- No data lineage tracking / experiment tracking has been implemented from waikey's end haha
- Please follow the steps below to reproduce the features

### Flow
1. Split Raw Data (the big csv gotten from Kaggle) into 5 parquets based on the variable type (2 for delinquency variables)
   * [single notebook](https://github.com/waikey-lee/kaggle-amex/blob/main/notebooks/0a.split_raw_data.ipynb)
   * I store the result in train_parquet and test_parquet which I have [gitignore](https://github.com/waikey-lee/kaggle-amex/blob/main/raw_data/.gitignore) them
2. Clean the variables manually, e.g. denoise (round down to 2 decimal places) => further binning => fix spike if applicable
   * The notebooks are all in folder *interim_data*, right now I'm using v3, v4 is coming as well (further fix on delinquency variables)
   * Run from 1a to 1f in this [v3](https://github.com/waikey-lee/kaggle-amex/tree/main/interim_data/v3) folder should get the cleaner version
3. Generate aggregation features from the cleaned data
   * The notebooks are all in folder *processed_data*, right now I'm using v4
   * Should be able to produce the agg features in the same folder by running the [feature generation notebook](https://github.com/waikey-lee/kaggle-amex/blob/main/processed_data/v4/feature_gen.ipynb)
4. Training, I was mainly using LGBM dart, can refer to [this notebook](https://github.com/waikey-lee/kaggle-amex/blob/main/experiments/2.lgbm_dart_1020/train.ipynb) for the sample script, for the sake of better boost, appreciate if you guys can work on other models, for example
   * RNN
   * Standard NN
   * XGBoost
   * CatBoost
   * Or any other fancy algorithm if applicable