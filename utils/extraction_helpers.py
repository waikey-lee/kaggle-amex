import numpy as np
import pandas as pd

def read_file(path='', usecols=None, sort=False, replace_negative_one=False, replace_negative127=True):
    # LOAD DATAFRAME
    if path.endswith(".parquet"):
        if usecols is not None:
            df = pd.read_parquet(path, columns=usecols)
        else:
            df = pd.read_parquet(path)
    elif path.endswith(".ftr"):
        df = pd.read_feather(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".pkl"):
        df = pd.read_pickle(path)
        if usecols is not None:
            for col in usecols:
                if col not in df.columns:
                    df[col] = np.nan
                    print(f"Column {col} is missing, set it as all NaN")
            df = df.loc[:, usecols]
    else:
        print("Unknown file format")
    if "S_2" in df.columns:
        df["S_2"] = pd.to_datetime(df["S_2"])
    if "customer_ID" not in df.columns and df.shape[0] in [5531451, 458913]:
        df = df.reset_index().rename(columns={"index": 'customer_ID'})
    if sort:
        if "S_2" in df.columns and 'customer_ID' in df.columns:
            df.S_2 = pd.to_datetime(df.S_2)
            df = df.sort_values(['customer_ID', 'S_2'])
        elif 'customer_ID' in df.columns:
            df = df.sort_values(['customer_ID'])
        else:
            print("Nothing to sort")
            pass
    if replace_negative127:
        df = df.replace(-127, np.nan)
    if replace_negative_one:
        if df.shape[1] >= 180:
            df = df.replace(-1, np.nan)
    print('Shape of data:', df.shape)
    return df