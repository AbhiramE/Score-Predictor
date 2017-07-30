import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import pickle


def get_features_and_labels(fileName):
    df = pd.read_csv(fileName)
    df = df[['balls', 'wickets', 'ground_average', 'pp_balls_left', 'target', 'total_overs', 'runs']]
    df_features = df.drop('runs', axis=1)
    df_label = df[['runs']]
    return df_features.as_matrix(), df_label.as_matrix()


# Get the best hyper-parameters using GridSearchCV
def hyperparameter_selection(train_x, train_y, param_grid, method):
    clf = GridSearchCV(method, param_grid=param_grid)
    clf.fit(train_x, train_y)
    return clf


# Regress on Lasso to predict the Classes for the test set
def regress(trainX, trainY, testX, testY, clf):
    clf.fit(trainX, trainY)
    y_predict = clf.predict(testX)

    with open('decision_tree.pickle', 'wb') as f:
        pickle.dump(clf, f)

    return math.sqrt(mean_squared_error(y_pred=y_predict, y_true=testY))


def pipeline(X_tr, Y_tr):
    X_train, X_validate, y_train, y_validate = train_test_split(X_tr, Y_tr, test_size=0.2)

    # Get the best hyperparameters on selected features
    param_grid = {
                  "min_samples_split": [2, 10, 20],
                  "max_depth": [None, 2, 5, 10],
                  "min_samples_leaf": [1, 5, 10],
                  "max_leaf_nodes": [None, 10, 50, 100, 1000],
                  }

    method = DecisionTreeRegressor()
    clf = hyperparameter_selection(X_train, y_train, param_grid, method)
    print "Pipeline Best Hyperparameter ", clf.best_params_['min_samples_split'], \
        clf.best_params_["max_depth"], clf.best_params_["min_samples_leaf"], clf.best_params_["max_leaf_nodes"]

    # Run the regressor on the selected features and best hyperparameter
    decisionTree = DecisionTreeRegressor(
                                   min_samples_split=clf.best_params_['min_samples_split'],
                                   max_depth=clf.best_params_["max_depth"],
                                   min_samples_leaf=clf.best_params_["min_samples_leaf"],
                                   max_leaf_nodes=clf.best_params_["max_leaf_nodes"])
    rmse = regress(X_train, y_train, X_validate, y_validate, decisionTree)
    print "Pipeline RMSE ", rmse


X_tr, y_tr = get_features_and_labels('../../Train_2nd_Innings.csv')
pipeline(X_tr, y_tr)
