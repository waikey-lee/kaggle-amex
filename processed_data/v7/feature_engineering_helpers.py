import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.feature_group import (
    CATEGORY_COLUMNS, NON_FEATURE_COLUMNS
)
from utils.eda_helpers import get_cols, insert_row_number
from utils.preprocess_helpers import clip_col

def filter_df_for_feature(df, cond_col, equal_to, rename_suffix, 
                          id_col="customer_ID", drop_cols=NON_FEATURE_COLUMNS):
    df_ = df.loc[df[cond_col] == equal_to]
    df_ = df_.set_index(id_col)
    df_ = df_.drop(columns=drop_cols, errors="ignore")
    df_ = df_.rename(columns={f: f"{f}_{rename_suffix}" for f in df_.columns})
    gc.collect()
    return df_

def get_specific_row_df(df):
    
    # Get last value for all features
    last_df = filter_df_for_feature(
        df, 
        cond_col="row_number", 
        equal_to=1, 
        rename_suffix="last"
    )
    print("Last entry done")
    gc.collect()
    
    # Get first value for all features
    first_df = filter_df_for_feature(
        df, 
        cond_col="row_number_inv", 
        equal_to=1, 
        rename_suffix="first"
    )
    print("First entry done")
    gc.collect()
    
#     # Get previous value (2nd last) for all features, there will be some missing (for those having only single statement)
#     second_last_df = filter_df_for_feature(
#         df, 
#         cond_col="row_number", 
#         equal_to=2, 
#         rename_suffix="second_last"
#     )
#     print("Second last entry done")
#     gc.collect()
    
#     # Get previous value (3rd last) for all features, there will be some missing (for those having only single statement)
#     third_last_df = filter_df_for_feature(
#         df, 
#         cond_col="row_number", 
#         equal_to=3, 
#         rename_suffix="third_last"
#     )
#     print("Third last entry done")
#     gc.collect()
    
    all_df = pd.concat(
        [
            last_df, 
            # second_last_df,
            # third_last_df,
            first_df,
        ], 
        axis=1
    )
    del df, last_df, first_df #, second_last_df, third_last_df
    return all_df

def get_agg_df(df):
    
    cid = 'customer_ID'
    cat_cols = set(df.select_dtypes("category").columns)
    numeric_columns = list(set(df.columns) - set(CATEGORY_COLUMNS) - set(NON_FEATURE_COLUMNS) - cat_cols)
    all_columns = list(set(numeric_columns).union(set(CATEGORY_COLUMNS)))
    
    avg_ = (df
      .groupby(cid)
      .mean()[numeric_columns]
      .rename(columns={f: f"{f}_avg" for f in numeric_columns})
    )
    print("Average done")
    gc.collect()
    
    min_ = (df
      .groupby(cid)
      .min()[numeric_columns] 
      .rename(columns={f: f"{f}_min" for f in numeric_columns})
    )
    print("Minimum done")
    gc.collect()
    
    max_ = (df
      .groupby(cid)
      .max()[numeric_columns]
      .rename(columns={f: f"{f}_max" for f in numeric_columns})
    )
    print("Maximum done")
    gc.collect()
    
    std_ = (df
      .groupby(cid)[numeric_columns]
      .std()
      .rename(columns={f: f"{f}_std" for f in numeric_columns})
    )
    print("Standard Deviation done")
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

def get_ma_df(df):
    cid = 'customer_ID'
    cat_cols = set(df.select_dtypes("category").columns)
    numeric_columns = list(set(df.columns) - set(CATEGORY_COLUMNS) - set(NON_FEATURE_COLUMNS) - cat_cols)
    all_columns = list(set(numeric_columns).union(set(CATEGORY_COLUMNS)))
    
    ma2_r1 = (df
      .loc[df["row_number"].between(1, 2)]
      .groupby(cid)[numeric_columns]
      .mean()
      .rename(columns={f: f"{f}_ma2_r1" for f in numeric_columns})
    )
    print("MA2 for Recency 1 done")
    gc.collect()
    
    ma2_r2 = (df
      .loc[df["row_number"].between(3, 4)]
      .groupby(cid)[numeric_columns]
      .mean()
      .rename(columns={f: f"{f}_ma2_r2" for f in numeric_columns})
    )
    print("MA2 for Recency 2 done")
    gc.collect()
    
    ma2_r3 = (df
      .loc[df["row_number"].between(5, 6)]
      .groupby(cid)[numeric_columns]
      .mean()
      .rename(columns={f: f"{f}_ma2_r3" for f in numeric_columns})
    )
    print("MA2 for Recency 3 done")
    gc.collect()
    
    ma3_r1 = (df
      .loc[df["row_number"].between(1, 3)]
      .groupby(cid)[numeric_columns]
      .mean()
      .rename(columns={f: f"{f}_ma3_r1" for f in numeric_columns})
    )
    print("MA3 for Recency 1 done")
    gc.collect()
    
    ma3_r2 = (df
      .loc[df["row_number"].between(4, 6)]
      .groupby(cid)[numeric_columns]
      .mean()
      .rename(columns={f: f"{f}_ma3_r2" for f in numeric_columns})
    )
    print("MA3 for Recency 2 done")
    gc.collect()
    
    ma3_first = (df
      .loc[df["row_number_inv"].between(1, 3)]
      .groupby(cid)[numeric_columns]
      .mean()
      .rename(columns={f: f"{f}_ma3_first" for f in numeric_columns})
    )
    print("MA3 for least Recency done")
    gc.collect()
    
    all_df = pd.concat(
        [
            ma2_r1,
            ma2_r2,
            ma2_r3,
            ma3_r1, 
            ma3_r2, 
            ma3_first,
        ], 
        axis=1
    )
    
    del df, ma2_r1, ma2_r2, ma2_r3, ma3_r1, ma3_r2, ma3_first
    return all_df

def feature_gen_pipeline(df):
    cat_columns = set(CATEGORY_COLUMNS).intersection(set(df.columns))
    df.loc[:, cat_columns] = df.loc[:, cat_columns].astype("category")
    insert_row_number(df)
    agg = get_agg_df(df)
    agg["num_statements"] = (
        df.loc[df["row_number"] == 1][["row_number", "row_number_inv"]].sum(axis=1) - 1
    ).reset_index(drop=True).values
    
    last_etc = get_specific_row_df(df)
    agg = last_etc.merge(agg, left_index=True, right_index=True, how="inner")
    del last_etc
    
    ma_df = get_ma_df(df)
    agg = agg.merge(ma_df, left_index=True, right_index=True, how="inner")
    del ma_df
    
    numeric_columns = list(set(df.columns) - set(CATEGORY_COLUMNS) - set(NON_FEATURE_COLUMNS))
    del df
    drop_columns = []
    for col in tqdm(numeric_columns):
        try:
            agg[f"{col}_ma2_r1_r2"] = agg[f"{col}_ma2_r1"] / agg[f"{col}_ma2_r2"].replace(0, 0.001)
            agg[f"{col}_ma2_r1_r3"] = agg[f"{col}_ma2_r1"] / agg[f"{col}_ma2_r3"].replace(0, 0.001)
            agg[f"{col}_general_trend"] = (agg[f"{col}_ma3_r1"] - agg[f"{col}_ma3_first"]) / np.log(agg["num_statements"] + 1)
            gc.collect()

            agg[f"{col}_range"] = agg[f"{col}_max"] - agg[f"{col}_min"]
            agg[f"{col}_displacement_ratio"] = agg[f"{col}_last"] / agg[f"{col}_first"].replace(0, 0.001)
            agg[f"{col}_last_minus_avg"] = agg[f"{col}_last"] - agg[f"{col}_avg"]
            agg[f"{col}_coef_var"] = agg[f"{col}_std"] / agg[f"{col}_avg"].replace(0, 0.001)
            agg[f"{col}_trend_index"] = agg[f"{col}_general_trend"] / agg[f"{col}_coef_var"].replace(0, 0.001)
            gc.collect()
        except:
            print(f"Skip col {col}")
    drop_columns.append("num_statements")
    keep_columns = list(set(agg.columns) - set(drop_columns))
    return agg, keep_columns

# Data format conversion logic
def convert_all(agg_df, tol=1e-4):
    float64_columns = agg_df.select_dtypes("float64").columns.tolist()
    for col in tqdm(float64_columns):
        temp = agg_df[col].astype(np.float32).values
        if (temp - agg_df[col]).abs().max() < tol:
            agg_df[col] = agg_df[col].astype(np.float32)
    return agg_df

# Clipping logic
def clip_all(agg_df, max_thr=1e3, min_thr=-1e3):
    # Clip upper bound
    max_ = agg_df.max()
    max_columns = max_[max_ > max_thr].index.tolist()
    for col in tqdm(max_columns):
        max_threshold1 = np.percentile(agg_df[col].dropna(), 99.9)
        max_threshold2 = np.percentile(agg_df[col].dropna(), 99)
        if max_threshold1 <= max_thr:
            agg_df = clip_col(agg_df, col, top_value=max_threshold1, add_new_col=False)
        elif max_threshold2 <= max_thr:
            agg_df = clip_col(agg_df, col, top_value=max_threshold2, add_new_col=False)
        else:
            agg_df = clip_col(agg_df, col, top_value=max_thr, add_new_col=False)
    
    # Clip lower bound
    min_ = agg_df.min()
    min_columns = min_[min_ < min_thr].index.tolist()
    for col in tqdm(min_columns):
        min_threshold1 = np.percentile(agg_df[col].dropna(), 0.1)
        min_threshold2 = np.percentile(agg_df[col].dropna(), 1)
        if min_threshold1 >= min_thr:
            agg_df = clip_col(agg_df, col, btm_value=min_threshold1, add_new_col=False)
        elif min_threshold2 >= min_thr:
            agg_df = clip_col(agg_df, col, btm_value=min_threshold2, add_new_col=False)
        else:
            agg_df = clip_col(agg_df, col, btm_value=min_thr, add_new_col=False)
    return agg_df
            
# Rounding logic
def round_all(agg_df, round_to=3, tol=1e-4):
    number_columns = agg_df.select_dtypes(np.number).columns.tolist()
    for col in tqdm(number_columns):
        temp = agg_df[col].round(round_to)
        if (temp - agg_df[col]).abs().max() < tol:
            agg_df[col] = agg_df[col].round(round_to)
    return agg_df