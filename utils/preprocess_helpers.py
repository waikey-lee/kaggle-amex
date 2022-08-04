import math
import numpy as np
from pandarallel import pandarallel

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
    
def round_dfs(train, test, col, decimals=2, add_new_col=True, nb_workers=8):
    if add_new_col:
        new_col = col + "_"
    else:
        new_col = col
    pandarallel.initialize(nb_workers=nb_workers, progress_bar=False, use_memory_fs=False)
    train[new_col] = train[col].parallel_apply(lambda x: round_decimals_down(x, decimals))
    test[new_col] = test[col].parallel_apply(lambda x: round_decimals_down(x, decimals))
    return train, test

def fill_nans(train, test, col, method="point", tuple_of_values=None, range_of_values=None, add_new_col=True):
    if add_new_col:
        new_col = col + "_"
    else:
        new_col = col
    train[new_col] = train[col].copy()
    test[new_col] = test[col].copy()
    if method == "point":
        train[new_col] = train[col].replace(tuple_of_values, np.nan)
        test[new_col] = test[col].replace(tuple_of_values, np.nan)
    elif method == "range":
        min_ = range_of_values[0]
        max_ = range_of_values[1]
        train.loc[train[col].between(min_, max_), new_col] = np.nan
        test.loc[test[col].between(min_, max_), new_col] = np.nan
    return train, test

def drop_temp_columns(train, test):
    init_len = int(train.shape[0])
    train = train.drop(columns=train.columns[train.columns.str.endswith("_")], errors="ignore")
    test = test.drop(columns=test.columns[test.columns.str.endswith("_")], errors="ignore")
    print(f"Drop {init_len - train.shape[0]} columns")
    return train, test

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