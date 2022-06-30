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
    else:
        print("Unknown file format")
    
    if "S_2" in df.columns:
        df.S_2 = pd.to_datetime(df.S_2)
        df = df.sort_values(['customer_ID', 'S_2'])
    else:
        df = df.sort_values(['customer_ID'])
    df = df.reset_index(drop=True)
    print('Shape of data:', df.shape)
    return df