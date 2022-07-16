import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.feature_group import (
    CATEGORY_COLUMNS, NON_FEATURE_COLUMNS, 
    MEAN_FEATURES, MIN_FEATURES, MAX_FEATURES, LAST_FEATURES,           # FIRST LEVEL AGG
    FIRST_FEATURES, RANGE_FEATURES, VELOCITY_FEATURES, SPEED_FEATURES   # SECOND LEVEL AGG
)

def filter_df_for_feature(df, cond_col, equal_to, rename_suffix, 
                          id_col="customer_ID", drop_cols=NON_FEATURE_COLUMNS):
    df_ = df.loc[df[cond_col] == equal_to]
    df_ = df_.set_index(id_col)
    df_ = df_.drop(columns=drop_cols, errors="ignore")
    df_ = df_.rename(columns={f: f"{f}_{rename_suffix}" for f in df_.columns})
    gc.collect()
    return df_

def get_specific_row_df(raw_df):
    
    df = raw_df.copy()
    
    # Get last value for all features
    last_df = filter_df_for_feature(
        df, 
        cond_col="row_number", 
        equal_to=1, 
        rename_suffix="last"
    )
    
    # Get first value for all features
    first_df = filter_df_for_feature(
        df, 
        cond_col="row_number_inv", 
        equal_to=1, 
        rename_suffix="first"
    )
    
    # Get previous value (2nd last) for all features, there will be some missing (for those having only single statement)
    second_last_df = filter_df_for_feature(
        df, 
        cond_col="row_number", 
        equal_to=2, 
        rename_suffix="second_last"
    )
    
    # Get previous value (2nd last) for all features, there will be some missing (for those having only single statement)
    third_last_df = filter_df_for_feature(
        df, 
        cond_col="row_number", 
        equal_to=3, 
        rename_suffix="third_last"
    )
    
    # Calculate aggregate features
    cid = pd.Categorical(df.pop('customer_ID'), ordered=True)
    
    all_df = pd.concat(
        [
            last_df, 
            second_last_df,
            third_last_df,
            first_df,
        ], 
        axis=1
    )
    del df, last_df, second_last_df, third_last_df, first_df
    return all_df

def get_agg_df(raw_df):
    df = raw_df.copy()
    
    # Calculate aggregate features
    cid = pd.Categorical(df.pop('customer_ID'), ordered=True)
    numeric_columns = list(set(df.columns) - set(CATEGORY_COLUMNS) - set(NON_FEATURE_COLUMNS))
    all_columns = list(set(numeric_columns).union(set(CATEGORY_COLUMNS)))
    
    avg_ = (df
      .groupby(cid)
      .mean()[numeric_columns]
      .rename(columns={f: f"{f}_avg" for f in numeric_columns})
    )
    gc.collect()
    
    min_ = (df
      .groupby(cid)
      .min()[numeric_columns] 
      .rename(columns={f: f"{f}_min" for f in numeric_columns})
    )
    gc.collect()
    
    max_ = (df
      .groupby(cid)
      .max()[numeric_columns]
      .rename(columns={f: f"{f}_max" for f in numeric_columns})
    )
    gc.collect()
    
    std_ = (df
      .groupby(cid)
      .std()[numeric_columns]
      .rename(columns={f: f"{f}_std" for f in numeric_columns})
    )
    gc.collect()
    
    all_df = pd.concat(
        [
            avg_, 
            min_, 
            max_, 
            std_
        ], 
        axis=1
    )
    
    del df, avg_, min_, max_, std_
    return all_df

def get_ma_df(raw_df):
    df = raw_df.copy()
    
    # Calculate aggregate features
    cid = "customer_ID"
    # cid = pd.Categorical(df.pop('customer_ID'), ordered=True)
    numeric_columns = list(set(df.columns) - set(CATEGORY_COLUMNS) - set(NON_FEATURE_COLUMNS))
    all_columns = list(set(numeric_columns).union(set(CATEGORY_COLUMNS)))
    
    ma3_r1 = (df
      .loc[df["row_number"].between(1, 3)]
      .groupby(cid)[numeric_columns]
      .mean()
      .rename(columns={f: f"{f}_ma3_r1" for f in numeric_columns})
    )
    gc.collect()
    
    ma3_r2 = (df
      .loc[df["row_number"].between(4, 6)]
      .groupby(cid)[numeric_columns]
      .mean()
      .rename(columns={f: f"{f}_ma3_r2" for f in numeric_columns})
    )
    gc.collect()
    
    ma3_r3 = (df
      .loc[df["row_number"].between(7, 9)]
      .groupby(cid)[numeric_columns]
      .mean()
      .rename(columns={f: f"{f}_ma3_r3" for f in numeric_columns})
    )
    gc.collect()
    
    ma3_r4 = (df
      .loc[df["row_number"].between(10, 12)]
      .groupby(cid)[numeric_columns]
      .mean()
      .rename(columns={f: f"{f}_ma3_r4" for f in numeric_columns})
    )
    gc.collect()
    
    all_df = pd.concat(
        [
            ma3_r1, 
            ma3_r2, 
            ma3_r3, 
            ma3_r4
        ], 
        axis=1
    )
    
    del df, ma3_r1, ma3_r2, ma3_r3, ma3_r4
    return all_df

def recursive_impute_using_knn(df, corr_df, corr_thr=0.3, corr_search_step_size=0.02, 
                               predictor_size_thr=5, list_of_k=[99], max_try_threshold=6, 
                               skip_first_n=0):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values()
    impute_columns = missing.index.tolist()

    for impute_column in impute_columns[skip_first_n:]:
        print(f"Selecting correlated column with {impute_column}...")
        curr_corr = corr_thr
        predictor_columns = []
        max_tries = 0
        while len(predictor_columns) < predictor_size_thr and max_tries < max_try_threshold:
            
            if curr_corr < corr_thr:
                print(f"Re-selecting correlated column using {curr_corr}")
            curr_corr -= corr_search_step_size
            max_tries += 1
            
            high_corr_columns = corr_df.loc[
                corr_df[impute_column].abs().between(curr_corr, 0.999), impute_column
            ].sort_values(ascending=False).index.tolist()
            no_missing_columns = df.isnull().sum()[df.isnull().sum() == 0].index.tolist()
            predictor_columns = list(set(high_corr_columns).intersection(set(no_missing_columns)))
            predictor_columns = predictor_columns[:predictor_size_thr]
        if max_tries >= max_try_threshold:
            print("Exceed max tries in searching correlated columns, skip this feature")
            continue
        train_val_knn = df.loc[~df[impute_column].isnull()]
        test_knn = df.loc[df[impute_column].isnull()]
        print(f"{predictor_columns} selected as predictors")
        if test_knn.shape[0] == 0:
            print(f"{impute_column} has no missing values, skip\n")
            continue
        train_knn, val_knn = train_test_split(train_val_knn, test_size=0.2, random_state=20)
        print(f"Train, Validation, Test size: {train_knn.shape[0], val_knn.shape[0], test_knn.shape[0]}")
        min_rmse = np.inf
        best_k = 0
        std = df[impute_column].std()
        print(f"{impute_column} standard deviation: {std:.4f}")
        for k in list_of_k:
            knn_model = KNeighborsRegressor(n_neighbors=k).fit(
                train_knn.loc[:, predictor_columns], 
                train_knn.loc[:, impute_column]
            )
            y_val_pred = knn_model.predict(val_knn.loc[:, predictor_columns])
            rmse = np.sqrt(mean_squared_error(val_knn.loc[:, impute_column], y_val_pred))
            print(f"K: {k}, Validation RMSE: {rmse:.5f}")
            if rmse < min_rmse:
                min_rmse = rmse
                best_knn_model = knn_model
                best_k = k
        print(f"Best K is {best_k}")
        if rmse >= std:
            print(f"Standard deviation smaller than RMSE, stop the imputation")
            continue
        df.loc[test_knn.index, impute_column] = best_knn_model.predict(test_knn.loc[:, predictor_columns])
        if df[impute_column].isnull().sum() > 0:
            print(f"Please check why column {impute_column} has yet to be imputed")
        print(f"Imputation done!\n")
        
    return df