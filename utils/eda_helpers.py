# This Python script contains some functions that I written before
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Get specific columns
def get_cols(df, keys, first=False):
    if isinstance(keys, str):
        keys = [keys]
    if not first:
        return [ col for col in df.columns if any(key in col for key in keys) ]
    else:
        return [ col for col in df.columns if any(col.startswith(key) for key in keys) ]

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

# Plot scatterplot
def plot_scatterplot(df, column, column2, hue_column):
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.scatterplot(data=df, x=column, y=column2, hue=hue_column, style=hue_column, 
                    palette="deep", s=7, legend="full")
    ax.set_title(f"Scatterplot of {column2} (y) against {column} (x)")
    ax.legend()
    plt.show()

# Plot count in bar chart & target rate in line chart
def groupby_catcol_plot(data, groupby_col, target_col="target", figsize=(15, 8), rotate90=False, 
                        labels=None, central_measure="mean", sort_by_count=False):
    temp = data.groupby(groupby_col).agg(target_central=(target_col, central_measure), 
                                         count=(target_col, "count")).reset_index()
    fig, ax1 = plt.subplots(figsize=figsize)
    if rotate90:
        plt.xticks(rotation=90)
    if sort_by_count:
        temp = temp.sort_values(by="count")
        sns.barplot(data=temp, x=groupby_col, y='count', alpha=0.5, ax=ax1)
        ax2 = ax1.twinx()
        sns.lineplot(data=temp["target_central"], marker='o', ax=ax2)
    else:
        temp = temp.sort_values(by="target_central")
        sns.lineplot(data=temp["target_central"], marker='o', ax=ax1)
        ax2 = ax1.twinx()
        sns.barplot(data=temp, x=groupby_col, y='count', alpha=0.5, ax=ax2)
    
    if labels is not None:
        ax2.set_xticklabels(labels)
    plt.title(f"{central_measure} of {target_col} (line) and count (bar) across column {groupby_col}")
    plt.show()