# This Python script contains some functions that I written before
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Get specific columns
def get_cols(df, keys, first=False, excludes=["xyz"]):
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(excludes, str):
        excludes = [excludes]
    if not first:
        return [ col for col in df.columns if any(key in col for key in keys) and all(exc not in col for exc in excludes) ]
    else:
        return [ col for col in df.columns if any(col.startswith(key) for key in keys) and all(exc not in col for exc in excludes) ]

# Check missing values
def check_missing_values(df, percent=True):
    null_df = df.isnull().sum()
    if percent:
        return (null_df[null_df > 0] / df.shape[0]) * 100
    else:
        return null_df[null_df > 0]
    
# Function to plot simple barchart
def plot_bar(df, cat_col, num_col="count", title=None, 
             horizontal=True, decimal=2, figsize=(15, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        sns.barplot(y=df[cat_col], x=df[num_col], ax=ax)
        for idx, value in enumerate(df[num_col]):
            display_value = round(value, decimal)
            plt.text(value, idx, display_value)
    else:
        sns.barplot(x=df[cat_col].astype(str), y=df[num_col], ax=ax)
        for idx, value in enumerate(df[num_col]):
            display_value = round(value, decimal)
            plt.text(idx, value, display_value)
    ax.set_title(title)
    plt.show()
    
# Function to plot simple barchart for missing data checking
def plot_missing_proportion_barchart(df, top_n=30, **kwargs):
    missing_prop_df = pd.DataFrame(check_missing_values(df).items(), 
                                   columns=["column", "missing_proportion"])
    missing_prop_df = missing_prop_df.sort_values(by="missing_proportion", 
                                                  ascending=False)
    plot_bar(missing_prop_df.iloc[:top_n], 
             cat_col="column",
             num_col="missing_proportion", 
             title="Missing Proportion in each columns",
             **kwargs)
    return missing_prop_df

# Check missing value x target distribution
def single_col_target_check(df, column, q=20):
    null_proportion = df.loc[df[column].isnull()]
    print(f"{null_proportion.shape[0]} null count, {null_proportion.shape[0] / df.shape[0]:.3f} null proportion")
    print(f"{null_proportion['target'].mean():.4f} of the targets have label = 1")
    
    if df[column].nunique() >= 30:
        df["temp"] = (pd.qcut(df[column], q=q).cat.codes + 1)
        temp_df = df.loc[df["temp"] > 0]
        title = f"Target distribution by {q} bins"
    else:
        df["temp"] = df[column].copy()
        temp_df = df.copy()
        title = "Target distribution by category"
    summary = pd.DataFrame(
        dict(
            target_mean=temp_df.groupby(["temp"])["target"].mean(), 
            count_distribution=temp_df["temp"].value_counts(),
            proportion_distribution=temp_df["temp"].value_counts(normalize=True)
        )
    ).reset_index()
    summary = summary.rename(columns={"index": column})
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax2 = ax.twinx()
    ax2.bar(summary[column], summary["count_distribution"], alpha=0.6)
    ax.plot(summary[column], summary["target_mean"], color="red")
    plt.title(title)
    plt.show()
    return summary

# Plot scatterplot
def plot_scatterplot(df, column, column2, hue_column=None):
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.scatterplot(data=df, x=column, y=column2, hue=hue_column, style=hue_column, 
                    palette="deep", s=7, legend="full")
    ax.set_title(f"Scatterplot of {column2} (y) against {column} (x)")
    if hue_column is not None:
        ax.legend()
    plt.show()