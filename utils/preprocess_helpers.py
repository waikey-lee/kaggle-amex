import math
import numpy as np
from pandarallel import pandarallel

# Split Test Set into 2 portion (Public vs Private)
def split_public_private(test):
    temp = test.groupby("customer_ID").agg(last_statement_month=("S_2", "max")).reset_index()
    temp["last_statement_month"] = temp["last_statement_month"].dt.to_period("M")
    temp["last_statement_month"].value_counts(normalize=True)
    
    public_cid_list = temp.loc[temp["last_statement_month"] == "2019-04", "customer_ID"].tolist()
    private_cid_list = temp.loc[temp["last_statement_month"] == "2019-10", "customer_ID"].tolist()
    
    public_test = test.loc[test["customer_ID"].isin(public_cid_list)].reset_index(drop=True)
    private_test = test.loc[test["customer_ID"].isin(private_cid_list)].reset_index(drop=True)
    print(f"Public size: {public_test.shape[0]}, Private size: {private_test.shape[0]}")
    return public_test, private_test

def round_decimals_down(number:float, decimals:int=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    try:
        return math.floor(number * factor) / factor
    except:
        return number

# Round all dataframe in df_list (Train, PB_Test, PV_Test)
def round_dfs(df_list, col, decimals=2, add_new_col=True, nb_workers=8):
    if add_new_col:
        new_col = col + "_"
    else:
        new_col = col
    pandarallel.initialize(nb_workers=nb_workers, progress_bar=False, use_memory_fs=False)
    for df in df_list:
        df[new_col] = df[col].parallel_apply(lambda x: round_decimals_down(x, decimals))
    return df_list

# Manual stack bar (to get a normal bell shape curve)
def manual_stack(x, start, stack_interval, denom):
    if x >= start:
        return start + np.floor((x - start) / stack_interval) / denom
    else:
        return x

def fill_nans(df_list, col, method="point", tuple_of_values=None, range_of_values=None, add_new_col=True):
    if add_new_col:
        new_col = col + "_"
    else:
        new_col = col
    for df in df_list:
        df[new_col] = df[col].copy()
        if method == "point":
            df[new_col] = df[col].replace(tuple_of_values, np.nan)
        elif method == "range":
            min_ = range_of_values[0]
            max_ = range_of_values[1]
            df.loc[df[col].between(min_, max_), new_col] = np.nan
    return df_list

def drop_temp_columns(df_list):
    init_len = int(df_list[0].shape[1])
    for df in df_list:
        df = df.drop(columns=df.columns[df.columns.str.endswith("_")].tolist(), errors="ignore")
    print(f"Drop {init_len - df.shape[1]} columns")
    return df_list

def manual_mapping(train, test, col, mapping_dict, add_new_col=True):
    if add_new_col:
        new_col = col + "_"
    else:
        new_col = col
    train[new_col] = np.where(train[col].isin(mapping_dict), train[col].map(mapping_dict), train[col])
    test[new_col] = np.where(test[col].isin(mapping_dict), test[col].map(mapping_dict), test[col])
    return train, test

def integerize(series, dtype=np.int32):
    return series.fillna(-127).astype(dtype)