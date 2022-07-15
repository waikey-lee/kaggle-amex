import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


def amex_metric(y_true: np.array, y_pred: np.array) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by descring prediction values
    indices = np.argsort(y_pred)[::-1]
    preds, target = y_pred[indices], y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d), g, d

def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(y_true, y_pred)[0],
            True)


# Plot ROC AUC curves for both train and test
def plot_roc_curves(legits_list, pred_probs_list, labels=["Train", "Test"], title=None):
    fig, ax = plt.subplots(figsize=(15, 8))
    for legits, probs, label in zip(legits_list, pred_probs_list, labels):
        fpr, tpr, thresholds  = roc_curve(legits, probs)
        auc = roc_auc_score(legits, probs)
        ax.plot(fpr, tpr, label=f"{label} AUC={auc:.4f}")
    if title is not None:
        plt.title(title)
    plt.legend(loc=4)
    plt.show()

# Plot Precision Recall curves for both train and test
def plot_precision_recall_curves(legits_list, pred_probs_list, labels=["Train", "Test"], title=None):
    fig, ax = plt.subplots(figsize=(15, 8))
    for legits, probs, label in zip(legits_list, pred_probs_list, labels):
        precision, recall, _ = precision_recall_curve(legits, probs)
        ap = average_precision_score(legits, probs)
        ax.plot(recall, precision, label=f"{label} Avg Precision={ap:.3f}")
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.legend()
    plt.show()

# Get the model evaluation metrics for training and test 
def get_performance_metrics(legits_list, preds_list, labels, metrics, functions):
    result_dict = dict()
    for legits, preds, label in zip(legits_list, preds_list, labels):
        result_dict[label] = [func(legits, preds) for func in functions]
    performance_df = pd.DataFrame(result_dict, index=metrics)
    return performance_df

# Plot feature importances
def plot_feature_importance(features, importances, title=None, limit=100, figsize=(15, 8), ascending=False):
    imp_df = pd.DataFrame(dict(feature=features, feature_importance=importances))
    imp_df = imp_df.sort_values(by="feature_importance", ascending=ascending).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=imp_df.iloc[:limit]["feature_importance"], 
                y=imp_df.iloc[:limit]["feature"], 
                ax=ax)
    if title is not None:
        plt.title(title)
    plt.show()
    return imp_df

# Main function use to evaluate model
def amex_metric_np(preds: np.ndarray, target: np.ndarray) -> float:
    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max
    return 0.5 * (g + d), g, d

# Get evaluation df
def get_final_metric_df(X: pd.DataFrame, y_true: pd.DataFrame, y_pred: pd.DataFrame):
    df = (pd.concat([X, y_true, y_pred], axis='columns')
          .sort_values('prediction', ascending=False))
    top_four_percent_df = df.copy()
    top_four_percent_df['weight'] = top_four_percent_df['target'].apply(lambda x: 20 if x==0 else 1)
    four_pct_cutoff = int(0.04 * top_four_percent_df['weight'].sum())
    top_four_percent_df['weight_cumsum'] = top_four_percent_df['weight'].cumsum()
    top_four_percent_df["is_cutoff"] = 0
    top_four_percent_df.loc[top_four_percent_df['weight_cumsum'] <= four_pct_cutoff, "is_cutoff"] = 1
    
    gini_df = df.copy()
    gini_df['weight'] = gini_df['target'].apply(lambda x: 20 if x==0 else 1)
    gini_df['random'] = (gini_df['weight'] / gini_df['weight'].sum()).cumsum()
    total_pos = (gini_df['target'] * gini_df['weight']).sum()
    gini_df['cum_pos_found'] = (gini_df['target'] * gini_df['weight']).cumsum()
    gini_df['lorentz'] = gini_df['cum_pos_found'] / total_pos
    gini_df['gini'] = (gini_df['lorentz'] - gini_df['random']) * gini_df['weight']
    
    return top_four_percent_df, gini_df
    
# # Main metric function, pandas version
# def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

#     def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
#         df = (pd.concat([y_true, y_pred], axis='columns')
#               .sort_values('prediction', ascending=False))
#         df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
#         four_pct_cutoff = int(0.04 * df['weight'].sum())
#         df['weight_cumsum'] = df['weight'].cumsum()
#         df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
#         return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
#     def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
#         df = (pd.concat([y_true, y_pred], axis='columns')
#               .sort_values('prediction', ascending=False))
#         df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
#         df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
#         total_pos = (df['target'] * df['weight']).sum()
#         df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
#         df['lorentz'] = df['cum_pos_found'] / total_pos
#         df['gini'] = (df['lorentz'] - df['random']) * df['weight']
#         return df['gini'].sum()

#     def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
#         y_true_pred = y_true.rename(columns={'target': 'prediction'})
#         return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

#     g = normalized_weighted_gini(y_true, y_pred)
#     d = top_four_percent_captured(y_true, y_pred)

#     return 0.5 * (g + d), g, d

# Get model prediction by providing model inputs, model and threshold
def calc_y_pred(X_test, model, threshold=50):
    y_pred_score = model.predict_proba(X_test)[:, 1]
    y_score_percentiles = [np.percentile(y_pred_score, i*5) for i in range(21)]
    exact_score_thresh = y_score_percentiles[threshold // 5]
    y_pred = (y_pred_score > exact_score_thresh).astype(int)
    return y_pred, exact_score_thresh

# Plot confusion matrix
def plot_confusion_matrix(df, metrics):
    plt.figure(figsize=(9, 7))
    sns.heatmap(df, cmap="coolwarm", annot=True, fmt=".0f")
    plt.show()

# Get model performance metrics for various threshold
def get_threshold_metrics_df(X_test, y_test, model):
    performance_metrics_list = []
    for threshold in range(1, 20):
        y_pred, score_thresh = calc_y_pred(X_test, model, threshold=threshold * 5)
        confusion_matrix, performance_metrics = calc_metrics(y_test, y_pred)
        performance_metrics["score_threshold_percentile"] = int(threshold * 5)
        performance_metrics["score_threshold"] = score_thresh
        performance_metrics_list.append(performance_metrics)
    metrics_df = pd.DataFrame(performance_metrics_list)
    return metrics_df

# Plot model performance metrics across thresholds
def plot_threshold_metrics(df, optim_metric=None, metrics=None, thresh_col="score_threshold"):
    if metrics is None:
        metrics = [col for col in df.columns if col != thresh_col]
    if optim_metric is None:
        optim_metric = metrics[0]
    optim_row = df.loc[df[optim_metric].argmax()]
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axvline(x=optim_row[thresh_col], linestyle="--", color="red")
    for col in metrics:
        try:
            ax.plot(df[thresh_col], df[col], label=col)
        except:
            print(f"Metric {col} can't found")
    
    ax.legend()
    plt.title("Performance metrics across threshold")
    plt.show()
    return dict(optim_row)

# Plot business metrics across thresholds
def plot_business_metrics(X_test, test, model):
    business_metrics_list, thresh_list = [], []
    for t in range(5, 100, 5):
        thresh = t / 100
        y_pred_test = (model.predict_proba(X_test)[:, 1] >= thresh).astype(int)
        bm = calc_business_metrics(test, y_pred_test)
        thresh_list.append(thresh)
        business_metrics_list.append(bm)
    
    result = pd.DataFrame(business_metrics_list)
    result["threshold"] = thresh_list
    optim_row = result.loc[result["Estimated revenue lift (times)"].argmax()]
    
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax1.axvline(x=optim_row["threshold"], linestyle="--", color="red")
    for col in result.drop(columns=["threshold", "Estimated revenue lift (times)"]):
        try:
            ax1.plot(result["threshold"], result[col], label=col)
        except:
            print(f"Metric {col} can't found")
    ax2 = ax1.twinx()
    ax2.plot(result["threshold"], 
             result["Estimated revenue lift (times)"], 
             label="Estimated revenue lift (times)", 
             color="red")
    ax1.legend()
    ax2.legend()
    plt.title("Business metrics across threshold")
    plt.show()
    return dict(optim_row)

# Self implemented base functions to calculate model performance metrics for binary classification
def calc_metrics(y_test, y_pred):
    result_table = pd.DataFrame(dict(ground_truth=y_test, 
                                     prediction=y_pred))
    gd_pos = (result_table["ground_truth"] == 1)
    pred_pos = (result_table["prediction"] == 1)
    TP = result_table.loc[gd_pos & pred_pos].shape[0]
    TN = result_table.loc[~gd_pos & ~pred_pos].shape[0]
    FP = result_table.loc[~gd_pos & pred_pos].shape[0]
    FN = result_table.loc[gd_pos & ~pred_pos].shape[0]
    confusion_matrix = np.array([[TP, FP], [FN, TN]])
    confusion_matrix = pd.DataFrame(confusion_matrix, columns=["Actual Positive", "Actual Negative"])
    confusion_matrix.index = ["Predicted Positive", "Predicted Negative"]
     
    accuracy = (TP + TN) / result_table.shape[0]
    if FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = np.nan
    if FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = np.nan
    f1 = 2 * precision * recall / (precision + recall)
     
    metrics = {"accuracy": accuracy, 
               "recall": recall, 
               "precision": precision, 
               "f1": f1}
     
    return confusion_matrix, metrics

# Get business metrics for various threshold
def calc_business_metrics(test, y_pred, columns=["total_pymnt", "loan_amnt"]):
    df = test.loc[:, columns]
    df["predicted_delinquent"] = y_pred
    pass_df = df.loc[df["predicted_delinquent"] == 0]
    pass_rate_upperbound = round(pass_df.shape[0] / df.shape[0], 3) * 100
    revenue_multiplier = (pass_df["total_pymnt"] - pass_df["loan_amnt"]).sum() / (df["total_pymnt"] - df["loan_amnt"]).sum()
    profit_margin = round((pass_df["total_pymnt"] - pass_df["loan_amnt"]).sum() / pass_df["loan_amnt"].sum(), 3) * 100
    business_metrics = {"Estimated pass rate": pass_rate_upperbound, 
                        "Estimated revenue lift (times)": revenue_multiplier, 
                        "Estimated profit margin": profit_margin}
    return business_metrics

# Putting everything together into a single function for model evaluation
def evaluation_full_suite(X_train, y_train, X_test, y_test, test, model, 
                          plot_metrics=["f1", "accuracy", "recall", "precision"]):
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    model_name = str(model).split("(")[0]
    plot_roc_curves(
        legits_list=[y_train, y_test], 
        pred_probs_list=[y_train_pred_proba, y_pred_proba], 
        labels=["Train", "Test"],
        title=f"ROC curves for train set and test set, {model_name} model")
    plot_precision_recall_curves(
        legits_list=[y_train, y_test], 
        pred_probs_list=[y_train_pred_proba, y_pred_proba], 
        labels=["Train", "Test"],
        title=f"Precision Recall Curves for Train set and Test set, {model_name} model"
    )
    model_metrics_df = get_threshold_metrics_df(X_test, y_test, model)
    _ = plot_threshold_metrics(model_metrics_df, metrics=plot_metrics)
    business_metrics = plot_business_metrics(X_test, test, model)
    optim_y_pred = (model.predict_proba(X_test)[:, 1] >= business_metrics["threshold"]).astype(int)
    cm, model_metrics = calc_metrics(y_test, optim_y_pred)
    plot_confusion_matrix(cm, model_metrics)
    all_metrics = {**model_metrics, **business_metrics}
    return all_metrics