import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import category_encoders as ce
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import lime
import lime.lime_tabular
import shap

from project_2.modeling.dataset import load_data
from project_2.modeling.features import nominal_vars
from project_2.modeling.features import ordinal_vars
from project_2.modeling.features import histgrams
from project_2.modeling.features import correlation_matrix
from project_2.modeling.train import transformers 
from project_2.modeling.train import split_data
from project_2.modeling.train import over_sampling
from project_2.modeling.train import train_model
from project_2.modeling.predict import make_predictions

def missing_data(df):
    matrix_missing = msno.matrix(df)
    plt.show()
    logging.info("Run missingness successfully")
    return matrix_missing

def histgrams(df):
    age = sns.histplot(df['age'])
    plt.show()
    balance = sns.histplot(df['balance'])
    plt.show()
    duration = sns.histplot(df['duration'])
    plt.show()
    day = sns.histplot(df['day'])
    logging.info("Histogram run successfully")
    plt.show()
    return age, balance, duration, day

def correlation_matrix(df):
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr))
    correlation_matrix = sns.heatmap(corr, center=0,
            mask=mask,
              linewidths=1,
              annot=True,
                fmt=".2f"
)
    plt.title("Correlation matrix")
    logging.info("Loading correlation successful")
    plt.show()
    return correlation_matrix


def feature_importance(model, X_train):
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    # A series is a better representation than using a dictionary 
    # index is needed for the labels
    top_features = importances.sort_values(ascending=False)
    # sort first for the importance, then sort again for the visual order of the va
    top_features.sort_values().plot(kind='barh', figsize=(12,6))
    # A horizontal bar chart is much better than a vertical bar chart to represent 
    plt.title("Feature Importance")
    logging.info("Top features visualized successfully")
    plt.show()

def explainable_shap_seen(model, X_train):
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent")
    shap_values = explainer(X_train)

    shap.plots.beeswarm(shap_values,
                        max_display=14,
                        color="coolwarm",
                        show= False)
    logging.info("Shap visual for seen data loaded successfully")
    plt.show()

def explainable_shap_unseen(model, X_test):
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent")
    shap_values = explainer(X_test)

    shap.plots.beeswarm(shap_values,
                        max_display=14,
                        color="coolwarm",
                        show= False)
    logging.info("Shap visual for unseen data loaded successfully")
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_data()
    nominal_vars(df)
    ordinal_vars(df)
    missing_data(df)
    histgrams(df)
    correlation_matrix(df)
    split_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    transformers(X_train, X_test)
    over_sampling(X_train, y_train)


    xgboost_tuned = XGBClassifier(random_state =42,
                         scale_pos_weight = 25,
                           max_depth = 35, 
                           learning_rate = 0.6)
    trained_xgboost_tuned = train_model(xgboost_tuned, X_train, y_train)
    xgboost_predications = make_predictions(trained_xgboost_tuned,X_test)
    feature_importance(trained_xgboost_tuned, X_train)
    explainable_shap_unseen(trained_xgboost_tuned, X_test)
    explainable_shap_seen(trained_xgboost_tuned, X_train)

# python -m project_2.modeling.plots


