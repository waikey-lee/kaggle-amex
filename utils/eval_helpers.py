import catboost
import joblib
import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from tqdm import tqdm

# Function to calculate overall amex metric, Gini, and 4% Positive capture rate
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

# Function to be use in LGBM in-train validation
def lgb_amex_metric(y_pred, train_data):
    """The competition metric with lightgbm's calling convention"""
    y_true = train_data.get_label()
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
    df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
    four_pct_cutoff = int(0.04 * df['weight'].sum())
    df['weight_cumsum'] = df['weight'].cumsum()
    df["is_cutoff"] = 0
    df.loc[df['weight_cumsum'] <= four_pct_cutoff, "is_cutoff"] = 1
    df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
    total_pos = (df['target'] * df['weight']).sum()
    df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
    df['lorentz'] = df['cum_pos_found'] / total_pos
    df['gini'] = (df['lorentz'] - df['random']) * df['weight']
    return df
    
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

class TreeExperiment:
    def __init__(self, exp_full_path, seed=None):
        self.path = exp_full_path
        self.target = []
        self.seed = seed
        self.models = {}
        self.feature_names = {}
        self.feature_importances = {}
        self.val_metrics = [0, 0, 0]
        self.master_feature_set = set()
        self.feature_imp_df = pd.DataFrame()
        self.feature_imp_summary = pd.DataFrame()
        self.read_models()
        self.retrieve_features()
        self.get_master_feature_set()
        self.get_feature_importance_summary()
    
    # To read all (usually 5) models from the experiment's directory
    def read_models(self):
        model_paths = [file for file in sorted(os.listdir(f"{self.path}/models")) if file.startswith("model")]
        for i, model_path in enumerate(model_paths):
            self.models[i] = joblib.load(f"{self.path}/models/{model_path}")
    
    # To retrieve the list of feature names, as well as their respective feature importances
    # Currently support only CatBoost Classifier & LGBM Booster
    def retrieve_features(self):
        for i, model in self.models.items():
            if type(model) == catboost.core.CatBoostClassifier:
                feature_name = self.models[i].feature_names_
                feature_imp = self.models[i].feature_importances_
            elif type(model) == lgb.basic.Booster:
                feature_name = self.models[i].feature_name()
                feature_imp = self.models[i].feature_importance()
            else:
                feature_name = self.models[i].feature_name_
                feature_imp = self.models[i].feature_importances_
            self.feature_names[i] = feature_name
            self.feature_importances[i] = feature_imp
    
    # Get the list of (union of) full feature list specfically for this experiment
    def get_master_feature_set(self):
        for feature_names in self.feature_names.values():
            self.master_feature_set = self.master_feature_set.union(feature_names)
    
    # Inference on training data, return full input + additional column of prediction score
    def inference_on_train(self, train):
        # Use the same seed during training to ensure inference only on held-out dataset
        if self.seed is not None:
            kf = StratifiedKFold(n_splits=5, random_state=self.seed, shuffle=True)
        else:
            kf = StratifiedKFold(n_splits=5)
        target = train["target"].values
        val_indices = [idx_va for idx_tr, idx_va in kf.split(train, target)]
        val_auc_list = []
        train["prediction"] = 0
        for model, idx_va, feature_list in tqdm(zip(self.models.values(), val_indices, self.feature_names.values())):
            temp = model.predict(
                train.loc[idx_va, feature_list], 
                raw_score=True
            )
            val_auc_list.append(roc_auc_score(
                train.loc[idx_va, "target"].values, 
                temp
            ))
            train.loc[idx_va, "prediction"] = temp
        return train, val_auc_list
    
    # Calculate Amex metric based on the train data with prediction score column
    def calc_validation_performance(self, train, target_col="target", prediction_col="prediction"):
        # General Preparation
        target = train[target_col].values
        all_pos = np.sum(target)
        train = train.sort_values(by=prediction_col, ascending=False)
        train['weight'] = train[target_col].apply(lambda x: 20 if x==0 else 1)
        
        # Calculate top 4 percent default capture rate
        four_pct_cutoff = int(0.04 * train['weight'].sum())
        train['weight_cumsum'] = train['weight'].cumsum()
        train["is_cutoff"] = 0
        train.loc[train['weight_cumsum'] <= four_pct_cutoff, "is_cutoff"] = 1
        # train = train.reset_index()
        top4_pos = train.loc[(train[target_col] == 1) & (train["is_cutoff"] == 1)].shape[0]
        d = top4_pos / all_pos
        
        # Calculate Gini
        train['random'] = (train['weight'] / train['weight'].sum()).cumsum()
        total_pos = (train[target_col] * train['weight']).sum()
        train['cum_pos_found'] = (train[target_col] * train['weight']).cumsum()
        train['lorentz'] = train['cum_pos_found'] / total_pos
        train['gini'] = (train['lorentz'] - train['random']) * train['weight']
        gini = train["gini"].sum()
        all_neg = target.shape[0] - all_pos
        gini_max = 10 * all_neg * (all_pos + 20 * all_neg - 19) / (all_pos + 20 * all_neg)
        g = gini / gini_max
        
        return train, (0.5 * (g + d), g, d)
    
    # Run both inference on train & calculate validation performance
    def get_validation_performance(self, train):
        train, val_auc_list = self.inference_on_train(train)
        train, final_metrics = self.calc_validation_performance(train)
        return train, final_metrics, val_auc_list
    
    # Inference on whole dataset, by batch
    def inference_full(self, data, batch_size=5000):
        scores_list = []
        j = 0
        for model, feature_list in zip(self.models.values(), self.feature_names.values()):
            score_list = []
            j += 1
            print(f"Model {j}")
            for i in tqdm(range(int(data.shape[0] / batch_size) + 1)):
                score_list.append(model.predict(
                    data.loc[int(i * batch_size): int((i+1) * batch_size) - 1, feature_list], 
                    raw_score=True
                ))
            scores_list.append(np.concatenate(score_list))
        score_df = pd.DataFrame(np.stack(scores_list).T, columns=[f"score{i}" for i in range(1, 6)])
        return score_df
    
    @staticmethod
    def get_agg_type(df):
        if "feature" in df.columns:
            return df["feature"].str.split("_").str[2:].str.join("_").values
        else:
            return df.index.str.split("_").str[2:].str.join("_").values
    
    @staticmethod
    def get_base_feature_column(df):
        if "feature" in df.columns:
            return df["feature"].str.split("_").str[:2].str.join("_").values
        else:
            return df.index.str.split("_").str[:2].str.join("_").values
    
    def get_feature_importance_summary(self):
        feature_imps = []
        for i, model in self.models.items():
            feature_imps.append(
                pd.DataFrame(
                    {
                        "feature": self.feature_names[i], 
                        f"importance{i}": self.feature_importances[i]
                    }
                ).set_index("feature")
            )
        feature_imp_df = pd.concat(feature_imps, axis=1)
        feature_imp_df["average_importance"] = feature_imp_df.mean(axis=1)
        feature_imp_df = feature_imp_df.reset_index()
        feature_imp_df = feature_imp_df.sort_values("average_importance", ascending=False)
        
        feature_imp_df["agg_type"] = self.get_agg_type(feature_imp_df)
        feature_imp_df["base_feature"] = self.get_base_feature_column(feature_imp_df)
        self.feature_imp_df = feature_imp_df.copy()
        del feature_imp_df
        
        pivoted_feature_imp_df = pd.pivot_table(
            self.feature_imp_df, 
            values="average_importance", 
            index="base_feature", 
            columns="agg_type"
        ).drop(columns="", errors="ignore").reset_index()
        
        self.feature_imp_summary = pivoted_feature_imp_df.loc[
            (pivoted_feature_imp_df["base_feature"].str.contains("_")) & 
            (pivoted_feature_imp_df["base_feature"].str.len() < 10)
        ]