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
    
    del avg_, min_, max_, std_
    return all_df

def process_data(df):
    last_df = df.loc[df["row_number"] == 1]
    last_df = last_df.set_index("customer_ID")
    last_df = last_df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    last_df = last_df.rename(columns={f: f"{f}_last" for f in last_df.columns})
    gc.collect()
    
    first_df = df.loc[df["row_number_inv"] == 1]
    first_df = first_df.set_index("customer_ID")
    first_df = first_df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    first_df = first_df.rename(columns={f: f"{f}_first" for f in first_df.columns})
    gc.collect()
    
    previous_df = df.loc[df["row_number"] == 2]
    previous_df = previous_df.set_index("customer_ID")
    previous_df = previous_df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    previous_df = previous_df.rename(columns={f: f"{f}_previous" for f in previous_df.columns})
    gc.collect()
    
    cid = pd.Categorical(df.pop('customer_ID'), ordered=True)
    last = (cid != np.roll(cid, -1)) # mask for last statement of every customer
    first = (cid != np.roll(cid, 0))
    numeric_columns = list(set(df.columns) - set(CATEGORY_COLUMNS) - set(NON_FEATURE_COLUMNS))
    all_columns = list(set(numeric_columns).union(set(CATEGORY_COLUMNS)))
    
    avg = (df
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
    
    all_df = pd.concat(
        [
            avg, 
            min_, 
            max_, 
            last_df, 
            first_df,
            previous_df,
        ], 
        axis=1
    )
    
    del avg, min_, max_, last_df, first_df, previous_df
    for col in tqdm(numeric_columns):
        all_df[f"{col}_range"] = all_df[f"{col}_max"] - all_df[f"{col}_min"]
        all_df[f"{col}_speed"] = all_df[f"{col}_last"] - all_df[f"{col}_first"]
        all_df[f"{col}_lag1_diff"] = all_df[f"{col}_last"] - all_df[f"{col}_previous"]
        all_df[f"{col}_last_lift"] = all_df[f"{col}_last"] - all_df[f"{col}_avg"]
        all_df[f"{col}_velocity"] = all_df[f"{col}_speed"] / (all_df[f"{col}_range"] + all_df[f"{col}_avg"])
    return all_df