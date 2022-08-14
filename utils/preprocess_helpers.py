import math
import matplotlib.pyplot as plt
import numpy as np
from pandarallel import pandarallel
from utils.eda_helpers import check_psi

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

# Clip column based on percentile or exact value
def clip_col(df, col, percentile=False, top_pct=1, btm_pct=1, top_value=None, btm_value=None, add_new_col=True):
    if add_new_col:
        new_col = col + "_"
    else:
        new_col = col
    if percentile:
        top_value = np.percentile(df[col].dropna(), top_pct)
        btm_value = np.percentile(df[col].dropna(), btm_pct)
    if top_value is not None:
        df[new_col] = np.where(df[col] > top_value, top_value, df[col])
        col = new_col
    if btm_value is not None:
        df[new_col] = np.where(df[col] < btm_value, btm_value, df[col])
    return df

# Round down decimal as the noise in Amex is being added
def round_decimals_down(number:float, decimals:int=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        try:
            return math.floor(number)
        except:
            return number
        
    factor = 10 ** decimals
    try:
        return math.floor(number * factor) / factor
    except:
        return number

# Round all dataframe in df_list (Train, PB_Test, PV_Test)
def round_dfs(df_list, col, decimals=2, add_new_col=True, nb_workers=8):
    print(f"Before round, Train-Private PSI = {check_psi(df_list, col)[1]:.4f}")
    if add_new_col:
        new_col = col + "_"
    else:
        new_col = col
    pandarallel.initialize(nb_workers=nb_workers, progress_bar=False, use_memory_fs=False)
    for df in df_list:
        df[new_col] = df[col].parallel_apply(lambda x: round_decimals_down(x, decimals))
    print(f"After round, Train-Private PSI = {check_psi(df_list, new_col)[1]:.4f}")
    return df_list

# Help to check if binning has been done correctly (via scatterplot)
def check_binning(df, col, start=0, end=500):
    temp_col = col + "_"
    df[[col, temp_col]].drop_duplicates().sort_values(by=col).iloc[start:end].plot.scatter(x=col, y=temp_col)
    plt.show()

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