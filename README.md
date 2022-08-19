## Kaggle Amex Default Prediction Competition
### Intro
- No data lineage tracking / experiment tracking has been implemented from waikey's end haha
- Please follow the steps below to reproduce the features

### Flow
1. Split Raw Data (the big csv gotten from Kaggle) into 5 parquets based on the variable type (2 for delinquency variables)
   * [single notebook](https://github.com/waikey-lee/kaggle-amex/blob/main/notebooks/0a.split_raw_data.ipynb)
   * I store the result in train_parquet and test_parquet which I have [gitignore](https://github.com/waikey-lee/kaggle-amex/blob/main/raw_data/.gitignore) them
2. Clean the variables manually
   * An example of the flow: Denoise (round down to 2 decimal places) => further binning => fix spike if applicable
   * The notebooks are all in folder *interim_data*
     * v0 - interger dtype parquet clean by others
     * v1 - Round down and simple cleaning
     * v2 - Similar to v1, but created a few features from base, with suffix **_a**, aim for better monotone relationship
     * v3 - Started to fix the suspicious spike in some of the base features, fix all except delinquency variables
     * v4 - Similar to v3, but went through all variables once
     * v5 - First version with clipping, rounding on all columns
     * v6 - WIP, hope to reduce over-clipping
   * Run from 1a to 1f in this [v5](https://github.com/waikey-lee/kaggle-amex/tree/main/interim_data/v5) folder should get the clipped version
3. Generate aggregation features from the cleaned data
   * The notebooks are all in folder *processed_data*, right now I'm using [v6](https://github.com/waikey-lee/kaggle-amex/tree/main/processed_data/v6)
     * Based on previous experiment, [v2](https://github.com/waikey-lee/kaggle-amex/tree/main/processed_data/v2) is the most stable right now
   * Should be able to produce the agg features in the same folder by running the feature_gen notebook in the respective version folder
4. Training, I was mainly using LGBM dart, can refer to [this notebook](https://github.com/waikey-lee/kaggle-amex/blob/main/experiments/2.lgbm_dart_1020/train.ipynb) for the sample script, for the sake of better boost, we should try our other algorithms
   * RNN - This is giving a boost currently
   * Standard NN / Linear Model - ToDo: Train on the residuals?
   * XGBoost - ToDo: Train using monotone features
   * CatBoost - ToDo: Train using more categorical features (after feature interaction crossing)
