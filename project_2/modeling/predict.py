import logging
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from project_2.modeling.dataset import load_data
from project_2.modeling.train import nominal_vars
from project_2.modeling.train import ordinal_vars
from project_2.modeling.train import split_data
from project_2.modeling.train import transformers
from project_2.modeling.train import over_sampling
from project_2.modeling.train import train_model

def make_predictions(model, X_test):
    logging.info("Making predictions")
    predictions = model.predict(X_test)
    logging.info("Predictions successful")
    return predictions


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
    knn_predictions = make_predictions(trained_knn, X_test)

    logreg = LogisticRegression(random_state =42)
    trained_logreg = train_model(logreg, X_train, y_train)
    logreg_predictions = make_predictions(trained_logreg, X_test)

    svm = SVC(random_state =42)
    trained_svm = train_model(svm, X_train, y_train)
    svm_predictions = make_predictions(trained_svm, X_test)

    normal_tree = DecisionTreeClassifier(random_state=42,
                                     class_weight='balanced')
    trained_normal_tree = train_model(normal_tree, X_train, y_train)
    tree_predictions = make_predictions(trained_normal_tree,X_test)

    xgboost = XGBClassifier(random_state =42)
    trained_xgboost = train_model(xgboost,X_train,y_train)
    xgboost_predications = make_predictions(trained_xgboost,X_test)

    rf = RandomForestClassifier(random_state=42,  
                            class_weight='balanced')
    trained_rf = train_model(rf,X_train,y_train)
    rf_predictions = make_predictions(trained_rf, X_test)

    xgboost_tuned = XGBClassifier(random_state =42,
                         scale_pos_weight = 25,
                           max_depth = 35, 
                           learning_rate = 0.6)
    trained_xgboost_tuned = train_model(xgboost_tuned, X_train, y_train)
    xgboost_predications = make_predictions(trained_xgboost_tuned,X_test)

    #python -m project_2.modeling.predict