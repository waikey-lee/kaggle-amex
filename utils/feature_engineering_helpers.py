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
    
    # Get previous value (2nd last) for all features, there will be some missing (for those having only single statement)
    second_last_df = filter_df_for_feature(
        df, 
        cond_col="row_number", 
        equal_to=2, 
        rename_suffix="second_last"
    )
    print("Second last entry done")
    gc.collect()
    
    # Get previous value (2nd last) for all features, there will be some missing (for those having only single statement)
    third_last_df = filter_df_for_feature(
        df, 
        cond_col="row_number", 
        equal_to=3, 
        rename_suffix="third_last"
    )
    print("Third last entry done")
    gc.collect()
    
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

def get_agg_df(df):
    # df = raw_df.copy()
    
    # Calculate aggregate features
    # cid = pd.Categorical(df['customer_ID'], ordered=True)
    cid = 'customer_ID'
    numeric_columns = list(set(df.columns) - set(CATEGORY_COLUMNS) - set(NON_FEATURE_COLUMNS))
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
    
#     skew_ = (df
#       .groupby(cid)[numeric_columns]
#       .skew()
#       .rename(columns={f: f"{f}_skew" for f in numeric_columns})
#     )
#     print("Skewness done")
#     gc.collect()
    
    all_df = pd.concat(
        [
            avg_, 
            min_, 
            max_, 
            std_,
            # skew_
        ], 
        axis=1
    )
    
    del df, avg_, min_, max_, std_, # skew_
    return all_df

def get_ma_df(df):
    cid = 'customer_ID'
    numeric_columns = list(set(df.columns) - set(CATEGORY_COLUMNS) - set(NON_FEATURE_COLUMNS))
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
    
    ma2_first = (df
      .loc[df["row_number_inv"].between(1, 2)]
      .groupby(cid)[numeric_columns]
      .mean()
      .rename(columns={f: f"{f}_ma2_first" for f in numeric_columns})
    )
    print("MA2 for least Recency done")
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
    
    all_df = pd.concat(
        [
            ma2_r1,
            ma2_r2,
            ma2_r3,
            ma2_first,
            ma3_r1, 
            ma3_r2, 
        ], 
        axis=1
    )
    
    del df, ma2_r1, ma2_r2, ma2_r3, ma2_first, ma3_r1, ma3_r2
    return all_df

def feature_gen_pipeline(df):
    cat_columns = set(CATEGORY_COLUMNS).intersection(set(df.columns))
    df.loc[:, cat_columns] = df.loc[:, cat_columns].astype("category")
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
    for col in tqdm(numeric_columns):
        agg[f"{col}_range"] = agg[f"{col}_max"] - agg[f"{col}_min"]
        agg[f"{col}_displacement"] = agg[f"{col}_last"] - agg[f"{col}_first"]
        agg[f"{col}_last_first_ratio"] = agg[f"{col}_last"] / agg[f"{col}_first"]
        agg[f"{col}_velocity"] = agg[f"{col}_displacement"] / agg["num_statements"]
        agg[f"{col}_sprint"] = agg[f"{col}_last"] - agg[f"{col}_second_last"]
        agg[f"{col}_previous_sprint"] = agg[f"{col}_second_last"] - agg[f"{col}_third_last"]
        agg[f"{col}_acceleration"] = (agg[f"{col}_sprint"] / (agg[f"{col}_previous_sprint"] * agg[f"{col}_std"])).replace(
            [np.inf, -np.inf], np.nan
        )
        agg[f"{col}_last_minus_avg"] = agg[f"{col}_last"] - agg[f"{col}_avg"]
        agg[f"{col}_coef_var"] = (agg[f"{col}_std"] / agg[f"{col}_avg"]).replace([np.inf, -np.inf], np.nan)
        agg[f"{col}_ma3_r1_r2"] = agg[f"{col}_ma3_r1"] / agg[f"{col}_ma3_r2"]
        agg[f"{col}_ma2_r1_r2"] = agg[f"{col}_ma2_r1"] / agg[f"{col}_ma2_r2"]
        agg[f"{col}_ma2_r1_r3"] = agg[f"{col}_ma2_r1"] / agg[f"{col}_ma2_r3"]
        agg[f"{col}_general_trend"] = 100 * (agg[f"{col}_ma2_r1"] - agg[f"{col}_ma2_first"]) / agg["num_statements"]
        gc.collect()
    
    if "num_statements" in agg.columns:
        agg = agg.drop(columns=["num_statements"])
    # if "customer_ID" not in agg.columns:
    #     agg = agg.reset_index().rename(columns={"index": "customer_ID"})
    return agg