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
    if "customer_ID" not in df.columns:
        df = df.reset_index()
    if "index" in df.columns:
        df = df.rename(columns={"index": 'customer_ID'})
    if "S_2" in df.columns:
        df.S_2 = pd.to_datetime(df.S_2)
        df = df.sort_values(['customer_ID', 'S_2'])
    else:
        df = df.sort_values(['customer_ID'])
    print('Shape of data:', df.shape)
    return df