# This Python script contains some functions that I written before
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
from cycler import cycler
from IPython.display import display
from colorama import Fore, Back, Style
from scipy.stats import chi2_contingency

# Filter dataframe by statement number
def filter_df(df, which_statement=1, id_col="row_number"):
    if isinstance(which_statement, int):
        return df.loc[df[id_col] == which_statement]

# Print percentile
def print_percentile(df_list, col, percentile):
    name_list = ["train", "public test", "private test"]
    print(f"{percentile}th percentile:")
    for name, df in zip(name_list, df_list):
        print(name, ":", np.percentile(df[col].dropna(), percentile))
    
# Check summary df
def check_summary(df, col):
    check_df = df.groupby("customer_ID")[col].agg(["max", "min", "mean", "std"]).reset_index()
    return check_df

# Return the summary stats for all datasets
def describe_all(df_list, col, name_list=["train", "public test", "private test"]):
    desc_list = []
    for df, name in zip(df_list, name_list):
        desc_list.append(df[col].describe().rename(index=name))
    summary = pd.concat(desc_list, axis=1)
    summary.loc["null_proportion", :] = [df[col].isnull().sum() / df.shape[0] for df in df_list]
    return summary

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
            plt.text(value, idx + 0.2, display_value)
    else:
        sns.barplot(x=df[cat_col].astype(str), y=df[num_col], ax=ax)
        for idx, value in enumerate(df[num_col]):
            display_value = round(value, decimal)
            plt.text(idx, value, display_value)
    ax.set_title(title)
    plt.show()
    
# Function to plot simple barchart for missing data checking
def plot_missing_proportion_barchart(df, top_n=30, show_plot=True, return_df=True, **kwargs):
    missing_prop_df = pd.DataFrame(check_missing_values(df).items(), 
                                   columns=["column", "missing_proportion"])
    missing_prop_df = missing_prop_df.sort_values(by="missing_proportion", 
                                                  ascending=False)
    if show_plot:
        plot_bar(missing_prop_df.iloc[:top_n], 
                 cat_col="column",
                 num_col="missing_proportion",
                 **kwargs)
    if return_df:
        return missing_prop_df

# Check missing value x target distribution
def plot_target_check(df, column, q=20, return_df=False, figsize=(18, 8), 
                      use_raw_bin=False, strfy_x=False, nunique_thr=100, 
                      drop_outlier=False, without_drop_tail=True, percentile_drop=1):
    null_proportion = df.loc[df[column].isnull()]
    print(f"{null_proportion.shape[0]} null count, {null_proportion.shape[0] / df.shape[0]:.3f} null proportion")
    print(f"{null_proportion['target'].mean():.4f} of the targets have label = 1")
    
    if df[column].nunique() >= nunique_thr:
        if not use_raw_bin:
            df["temp"] = pd.qcut(df[column], q=q, duplicates="drop").cat.codes
        else:
            df["temp"] = pd.qcut(df[column], q=q, duplicates="drop")
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
    if df[column].nunique() < nunique_thr and nunique_thr >= 100:
        if without_drop_tail:
            min_ = -np.inf
            max_ = np.inf
        else:
            print(f"Top & Bottom {percentile_drop}% are dropped from this chart")
            min_ = np.percentile(df[column].dropna(), percentile_drop)
            max_ = np.percentile(df[column].dropna(), 100 - percentile_drop)
        summary = summary.loc[summary[column].between(min_, max_)]
    
    if df[column].nunique() >= nunique_thr and use_raw_bin:
        adjusted_x_series = summary[column].apply(lambda x: pd.Interval(left=round(x.left, 4), right=round(x.right, 4))).astype(str)
    elif strfy_x:
        adjusted_x_series = summary[column].round(2).astype(str)
    else:
        adjusted_x_series = summary[column]
        
    plt.rcParams['axes.facecolor'] = '#0057b8' # blue
    plt.rcParams['axes.prop_cycle'] = cycler(color=['#ffd700'] + plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])
    plt.rcParams['text.color'] = 'w'
    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    ax2.bar(
        adjusted_x_series, 
        summary["proportion_distribution"], 
        color="maroon", 
        alpha=0.5
    )
    ax2.set_ylabel("Count Proportion (sum to 100)", color="maroon")
    ax.plot(
        adjusted_x_series, 
        summary["target_mean"], 
        color="orange"
    )
    ax.set_ylabel("Target Positive Proportion", color="orange")
    fig.autofmt_xdate(rotation=45)
    plt.title(title)
    plt.show()
    if return_df:
        return summary

# Plot scatterplot
def plot_scatterplot(df, column, column2, hue_column=None, figsize=(18, 10), ticksize=7, **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x=column, y=column2, hue=hue_column, style=hue_column, 
                    s=ticksize, legend="full", **kwargs) # palette="deep", 
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
    
def plot_train_test_distribution(df_list, col, figsize=(18, 8), q=100, is_category=False,
                                 nunique_thr=100, return_df=False, without_drop_tail=False):
    if is_category:
        d1 = calculate_psi(df_list[0][col].astype(float), df_list[1][col].astype(float))
        d2 = calculate_psi(df_list[0][col].astype(float), df_list[-1][col].astype(float))
    else:
        d1, d2 = check_psi(df_list, col)
    print(f"Train-Public PSI: {d1:.4f}, Train-Private PSI: {d2:.4f}")
    if is_category:
        for df in df_list:
            df[col] = df[col].astype(float)
    if len(df_list) == 3:
        color_list = ["yellow", "orange", "green"]
        name_list = ["Train", "PublicTest", "PrivateTest"]
    else:
        color_list = ["yellow", "green"]
        name_list = ["Train", "PrivateTest"]
    fig, ax = plt.subplots(figsize=figsize)
    if df_list[0][col].nunique() >= nunique_thr:
        for df, color in zip(df_list, color_list):
            df[col].plot.kde(color=color)
    else:
        many_series = []
        for df, name in zip(df_list, name_list):
            df["temp"] = df[col].copy()
            count_series = df["temp"].value_counts(normalize=True).rename(name)
            many_series.append(count_series)
        proportion_count_df = pd.concat(many_series, axis=1).sort_index().reset_index().rename(columns={"index": col})
        
        if df_list[0][col].nunique() < nunique_thr and nunique_thr >= 100:
            if without_drop_tail:
                min_ = -np.inf
                max_ = np.inf
            else:
                print("Bottom 1% and Top 1% are dropped from this chart")
                min_ = np.percentile(df_list[0][col].dropna(), 1)
                max_ = np.percentile(df_list[0][col].dropna(), 99)
            proportion_count_df = proportion_count_df.loc[proportion_count_df[col].between(min_, max_)]
        
        X_axis = np.arange(proportion_count_df.shape[0])
        if len(df_list) == 2:
            plt.bar(X_axis - 0.2, proportion_count_df[name_list[0]], 0.4, label=name_list[0], alpha=0.8, color=color_list[0])
            plt.bar(X_axis + 0.2, proportion_count_df[name_list[1]], 0.4, label=name_list[1], alpha=0.8, color=color_list[1])
        elif len(df_list) == 3:
            plt.bar(X_axis - 0.275, proportion_count_df[name_list[0]], 0.275, label=name_list[0], alpha=0.8, color=color_list[0])
            plt.bar(X_axis, proportion_count_df[name_list[1]], 0.275, label=name_list[1], alpha=0.8, color=color_list[1])
            plt.bar(X_axis + 0.275, proportion_count_df[name_list[2]], 0.275, label=name_list[2], alpha=0.8, color=color_list[2])
        plt.xticks(X_axis, proportion_count_df[col].round(2))
        plt.xlabel(col)
        plt.ylabel("Proportion of records")
        ax.set_title(f"Train Test Distribution for {col}")
        plt.legend()
    fig.autofmt_xdate(rotation=45)
    plt.show()
    if return_df:
        return proportion_count_df
    
def check_overlap_missing(df, col1, col2, n1=np.nan, n2=np.nan):
    col1_null_indices = df.loc[df[col1] == n1].index
    col2_null_indices = df.loc[df[col2] == n2].index
    if n1 != n1:
        col1_null_indices = df.loc[df[col1].isnull()].index
    if n2 != n2:
        col2_null_indices = df.loc[df[col2].isnull()].index
    print(f"{col1} missing count {len(col1_null_indices)}")
    print(f"{col2} missing count {len(col2_null_indices)}")
    print(f"Both {col1} & {col2} missing count {len(set(col1_null_indices).intersection(set(col2_null_indices)))}")
    
def plot_sampled_time_series(train, labels, column, sample_size, pad_values=-1, common_y_axis=True):
    plt.rcParams['axes.facecolor'] = '#ffffff'
    plt.rcParams['axes.prop_cycle'] = cycler(color=['#000000'] + plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])
    plt.rcParams['text.color'] = 'w'
    train_cid_list = train["customer_ID"].unique().tolist()
    row = int(sample_size / 4)
    fig, ax = plt.subplots(row + 1, 4, figsize=(24, int(row * 3)))
    ax = ax.ravel()
    temp = random.sample(range(len(train_cid_list)), sample_size)
    sampled_cid_list = [train_cid_list[i] for i in temp]
    if common_y_axis:
        min_ = train[column].min() - 0.1 * train[column].std()
        max_ = train[column].max() + 0.1 * train[column].std()
    for i in range(sample_size):
        ax_ = ax[i]
        cid_ = sampled_cid_list[i]
        array = train.loc[train["customer_ID"] == cid_, column].values
        array = np.pad(array, pad_width=(13 - len(array), 0), mode="constant", constant_values=(pad_values))
        target_ = labels.loc[labels["customer_ID"] == cid_, "target"].values[0]
        if target_ == 1:
            ax_.plot(range(13), array, color="red")
        else:
            ax_.plot(range(13), array, color="blue")
        if common_y_axis:
            ax_.set_ylim((min_, max_))
        ax_.set_title(cid_)
    plt.suptitle(column)
    plt.show()

def check_psi(df_list, col):
    train_public_psi = calculate_psi(df_list[0][col], df_list[1][col])
    train_private_psi = calculate_psi(df_list[0][col], df_list[-1][col])
    return train_public_psi, train_private_psi
    
def calculate_psi(expected, actual, buckettype='bins', buckets=1000, axis=0):
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)

# Count the number of unqiue and Nulls across Train, Public Test & Private Test
def generate_nunique_and_nulls(df_list):
    nunique_df = pd.DataFrame(dict(
        train_nunique=df_list[0].nunique(),
        public_test_nunique=df_list[1].nunique(),
        private_test_nunique=df_list[2].nunique()
    ))
    diff_nunique_indices = ~nunique_df[["train_nunique", "public_test_nunique", "private_test_nunique"]].eq(
        nunique_df.loc[:, "train_nunique"], 
        axis=0
    ).all(1)
    diff_nunique_df = nunique_df.loc[diff_nunique_indices]
    
    null_df = pd.DataFrame(dict(
        train_null=df_list[0].isnull().sum() / df_list[0].shape[0],
        public_test_null=df_list[1].isnull().sum() / df_list[1].shape[0],
        private_test_null=df_list[2].isnull().sum() / df_list[2].shape[0],
    ))
    diff_null_indices = ~null_df[["train_null", "public_test_null", "private_test_null"]].eq(
        null_df.loc[:, "train_null"], 
        axis=0
    ).all(1)
    diff_null_df = null_df.loc[diff_null_indices]
    return diff_nunique_df, diff_null_df

# def calc_chi_square_test_p_value(train, column1, column2, plot=False):
#     t = train.groupby([column1, column2]).agg(customer_count=("customer_ID", "count")).reset_index()
#     pivot = pd.pivot_table(t, index=column1, columns=column2, values="customer_count").fillna(0)
#     if plot:
#         plot_heatmap(pivot, annot=True, fmt=".0f")
#     stat, p, dof, expected = chi2_contingency(pivot)
#     return p

# for col1, col2 in combinations(binary_columns, 2):
#     p_value = calc_chi_square_test_p_value(train, col1, col2)
#     print(col1, col2, p_value)