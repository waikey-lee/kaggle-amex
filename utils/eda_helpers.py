# This Python script contains some functions that I written before
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
from cycler import cycler
from IPython.display import display
from colorama import Fore, Back, Style
plt.rcParams['axes.facecolor'] = '#0057b8' # blue
plt.rcParams['axes.prop_cycle'] = cycler(color=['#ffd700'] +
                                         plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])
plt.rcParams['text.color'] = 'w'

# Insert row number to indicate which credit card statement for each particular record
def insert_row_number(df):
    if "row_number_inv" not in df.columns:
        df.insert(1, "row_number_inv", df.groupby("customer_ID")["S_2"].rank(method="first", ascending=True).astype(int))
    if "row_number" not in df.columns:
        df.insert(1, "row_number", df.groupby("customer_ID")["S_2"].rank(method="first", ascending=False).astype(int))
    print("Done insertion")
    
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
def plot_target_check(df, column, q=20, return_df=False, figsize=(18, 8)):
    null_proportion = df.loc[df[column].isnull()]
    print(f"{null_proportion.shape[0]} null count, {null_proportion.shape[0] / df.shape[0]:.3f} null proportion")
    print(f"{null_proportion['target'].mean():.4f} of the targets have label = 1")
    
    if df[column].nunique() >= 100:
        df["temp"] = pd.qcut(df[column], q=q).cat.codes
        title = f"Target distribution by {q} bins"
    else:
        df["temp"] = df[column].copy()
        title = "Target distribution by category"
    summary = pd.DataFrame(
        dict(
            target_mean=df.groupby(["temp"])["target"].mean(), 
            count_distribution=df["temp"].value_counts(),
            proportion_distribution=df["temp"].value_counts(normalize=True)
        )
    ).reset_index()
    summary = summary.rename(columns={"index": column})
    
    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    ax2.bar(summary[column], summary["proportion_distribution"], color="red", alpha=0.6)
    ax2.set_ylabel("Count Proportion (sum to 100)", color="red")
    ax.plot(summary[column], summary["target_mean"], color="orange")
    ax.set_ylabel("Target Positive Proportion", color="orange")
    
    plt.title(title)
    plt.show()
    if return_df:
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
    
# Plot Heatmap
def plot_heatmap(df, figsize=(15, 8), annot=False, fmt='g'):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data=df, annot=annot, cmap="coolwarm", fmt=fmt)
    plt.show()
    
# Plot pie chart to display distribution (for integer column) between train & test
def plot_int_feature_distribution(train, test, col):
    # the same figure for both subplots
    fig, ax = plt.subplots(1, 2, figsize=(4, 3), dpi=144)

    # plot the first pie chart in ax1
    cts = train[col].value_counts(normalize=True).to_frame()
    ax[0].pie(cts[col])

    # plot the sencond pie chart in ax2
    cts = test[col].value_counts(normalize=True).to_frame()
    ax[1].pie(cts[col])

    plt.show()
    
def plot_train_test_distribution(train, test, col, figsize=(18, 8), q=100):
    fig, ax = plt.subplots(figsize=figsize)
    if train[col].nunique() >= 100:
        train[col].plot.kde(color='yellow')
        test[col].plot.kde(color='green')
        plt.show()
    else:
        train["temp"] = train[col]
        test["temp"] = test[col]
        train_series = train["temp"].value_counts(normalize=True).reset_index().rename(columns={"temp": "train_count"})
        test_series = test["temp"].value_counts(normalize=True).reset_index().rename(columns={"temp": "test_count"})
        proportion_count_df = train_series.merge(
            test_series,
            on="index",
            how="outer"
        ).rename(columns={"index": col}).sort_values(by=col).reset_index(drop=True)
        X_axis = np.arange(proportion_count_df.shape[0])
        plt.bar(X_axis - 0.2, proportion_count_df["train_count"], 0.4, label='Train')
        plt.bar(X_axis + 0.2, proportion_count_df["test_count"], 0.4, label='Test')
        plt.xticks(X_axis, proportion_count_df[col])
        plt.xlabel(col)
        plt.ylabel("Proportion of records")
        ax.set_title(f"Train Test Distribution for {col}")
        plt.legend()
        plt.show()
    
def check_overlap_missing(df, col1, col2, n1=np.nan, n2=np.nan):
    col1_null_indices = df.loc[df[col1] == n1].index
    col2_null_indices = df.loc[df[col2] == n2].index
    print(f"{col1} missing count {len(col1_null_indices)}")
    print(f"{col2} missing count {len(col2_null_indices)}")
    print(f"Both {col1} & {col2} missing count {len(set(col1_null_indices).intersection(set(col2_null_indices)))}")