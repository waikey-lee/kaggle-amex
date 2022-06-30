def create_has_col(df, col):
    has_col = f"has_{col}"
    df.loc[:, has_col] = 0
    df.loc[~df[col].isnull(), has_col] = 1
    return df

def create_sign_col(df, col):
    sign_col = f"{col}_sign"
    df[sign_col] = df[col].apply(lambda x: 0 if x == 0 else x / abs(x))
    return df

def apply_all_fillna(df):
    # Simple Fill NA with 0
    for col in ["D_87", "D_88", "B_39", "B_42"]:
        df[col] = df[col].fillna(0)  # .apply(lambda x: (abs(x) + x) / 2).fillna(0)
    # Create has column
    for col in ["D_110", "D_111", "D_132", "D_134", "D_135", "D_136", "D_137", "D_138", 
                "R_9"]:
        df = create_has_col(df, col=col)
    # Create sign column
    for col in ["B_39"]:
        df = create_sign_col(df, col=col)
    return df