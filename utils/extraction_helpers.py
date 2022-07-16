import numpy as np
import pandas as pd

def read_file(path='', usecols=None):
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
    else:
        print("Unknown file format")
    if "customer_ID" not in df.columns and df.shape[0] in [5531451, 458913]:
        df = df.reset_index().rename(columns={"index": 'customer_ID'})
    if "S_2" in df.columns and 'customer_ID' in df.columns:
        df.S_2 = pd.to_datetime(df.S_2)
        df = df.sort_values(['customer_ID', 'S_2'])
    elif 'customer_ID' in df.columns:
        df = df.sort_values(['customer_ID'])
    else:
        pass
    if df.shape[1] >= 180:
        df = df.replace(-1, np.nan)
    print('Shape of data:', df.shape)
    return df