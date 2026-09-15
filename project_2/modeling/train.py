import pandas as pd
import numpy as np
import logging
import category_encoders as ce

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from project_2.modeling.dataset import load_data

def nominal_vars(df):
    logging.info("Dealing with the nominal variables")
    df['default'] = df['default'].map({'yes': 1, 'no': 0})
    df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
    df['loan'] = df['loan'].map({'yes': 1, 'no': 0})
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    df["education"] = df["education"].replace("unknown", np.nan)
    df["job"] = df["job"].replace("unknown", np.nan)
    df["contact"] = df["contact"].replace("unknown", np.nan)
    df['job'] = df['job'].map({'management':0,'technician':1,'entrepreneur':2, 'blue-collar':3,
                'retired': 4, 'admin': 5,  'services': 6, 'self-employed': 7,
                'unemployed': 8,  'housemaid': 9, 'student': 10 })
    
    df['marital'] = df['marital'].map({'single':0,'married':1,'divorced':2})
    df['contact'] = df['contact'].map({'cellular': 0, 'telephone': 1})
    logging.info("Dealing with the nominal variables was successful")
    return df


def ordinal_vars(df):
    logging.info("Dealing with the ordinal variables")
    education_encoder= ce.OrdinalEncoder(cols=['education'],return_df=True,
                            mapping=[{'col':'education',
    'mapping':{'primary': 0,'secondary': 1,'tertiary': 2}}])
    
    month_encoder= ce.OrdinalEncoder(cols=['month'],return_df=True,
                            mapping=[{'col':'month',
    'mapping':{'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5,
            'jun': 6, 'jul': 7, 'aug': 8, 'oct': 10, 'nov': 11, 'dec': 12}}])
    df[['education']] = education_encoder.fit_transform(df[['education']]).astype('int')
        #[[]] used to indicate dataframe not series
    df[['month']] = month_encoder.fit_transform(df[['month']]).astype('int')
    logging.info("Dealing with the ordinal variables was successful")
    return df


def split_data(df):
    logging.info("Spliting data")
    y = df['y']
    X = df.drop('y',axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.2, random_state=42, stratify= y )
    logging.info("Spliting was successful")
    return X_train, X_test, y_train, y_test


def transformers(X_train, X_test):
    logging.info("Trasforming variables")
    numeric_vars = ['balance', 'age', 'duration']
    missing_vars = ['job','education','contact']

    scaler = StandardScaler()
    mode_imputer = SimpleImputer(strategy="most_frequent")


    X_train[numeric_vars] = scaler.fit_transform(X_train[numeric_vars])
    X_test[numeric_vars] = scaler.fit_transform(X_test[numeric_vars])

    X_train[missing_vars] = mode_imputer.fit_transform(X_train[missing_vars])
    X_test[missing_vars] = mode_imputer.transform(X_test[missing_vars])
    logging.info("transforming was successful")
    return X_train, X_test


def over_sampling(X_train, y_train):
    logging.info("Oversampling the data")
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    logging.info("Oversampling was successful")
    return X_train, y_train


def train_model(model, X_train, y_train):
    logging.info("Training the data")
    model.fit(X_train, y_train)
    logging.info("Training was succesful")
    return model









if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_data()
    nominal_vars(df)
    ordinal_vars(df)
    split_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    transformers(X_train, X_test)
    over_sampling(X_train, y_train)
    
    knn = KNeighborsClassifier(n_neighbors=20)
    trained_knn = train_model(knn, X_train, y_train)

    logreg = LogisticRegression(random_state =42)
    trained_logreg = train_model(logreg, X_train, y_train)

    svm = SVC(random_state =42)
    trained_svm = train_model(svm, X_train, y_train)

    normal_tree = DecisionTreeClassifier(random_state=42,
                                     class_weight='balanced')
    trained_normal_tree = train_model(normal_tree, X_train, y_train)

    xgboost = XGBClassifier(random_state =42)
    trained_xgboost = train_model(xgboost,X_train,y_train)

    rf = RandomForestClassifier(random_state=42,  
                            class_weight='balanced')
    trained_rf = train_model(rf,X_train,y_train)

    xgboost_tuned = XGBClassifier(random_state =42,
                         scale_pos_weight = 25,
                           max_depth = 35, 
                           learning_rate = 0.6)
    trained_xgboost_tuned = train_model(xgboost_tuned, X_train, y_train)
    


