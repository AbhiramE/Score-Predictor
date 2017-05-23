import math
import numpy as np
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def get_features_and_labels(fileName):
    df = pd.read_csv(fileName)
    df = df[['balls', 'wickets', 'ground_average', 'pp_balls_left', 'target', 'total_overs', 'runs']]
    df_features = df.drop('runs', axis=1)
    df_label = df[['runs']]
    return df_features.as_matrix(), df_label.as_matrix()


# Get the best hyper-parameters using GridSearchCV
def hyperparameter_selection(train_x, train_y, param_grid, method):
    train_y = np.ravel(train_y)
    clf = GridSearchCV(method, param_grid=param_grid)
    clf.fit(train_x, train_y)
    return clf


# Regress on Lasso to predict the Classes for the test set
def regress(trainX, trainY, testX, testY, clf):
    trainY = np.ravel(trainY)
    clf.fit(trainX, trainY)
    y_predict = clf.predict(testX)

    with open('random_forest.pickle', 'wb') as f:
        pickle.dump(clf, f)

    return math.sqrt(mean_squared_error(y_pred=y_predict, y_true=testY))


def pipeline(X_tr, Y_tr):
    X_train, X_validate, y_train, y_validate = train_test_split(X_tr, Y_tr, test_size=0.2)

    # Get the best hyperparameters on selected features
    param_grid = {"n_estimators": [250, 500, 1000, 2000],
                  "max_features": [3, 5],
                  "max_depth": [10, 20],
                  "min_samples_split": [2, 4],
                  "bootstrap": [True, False]}

    method = RandomForestRegressor()
    clf = hyperparameter_selection(X_train, y_train, param_grid, method)
    print "Pipeline Best Hyperparameter ", clf.best_params_['min_samples_split'], \
        clf.best_params_["max_depth"], clf.best_params_["min_samples_leaf"], clf.best_params_["max_leaf_nodes"], \
        clf.best_params_["bootstrap"]

    # Run the regressor on the selected features and best hyperparameter
    decisionTree = RandomForestRegressor()
    rmse = regress(X_train, y_train, X_validate, y_validate, decisionTree)
    print "Pipeline RMSE ", rmse

X_tr, y_tr = get_features_and_labels('../../Train_2nd_Innings.csv')
pipeline(X_tr, y_tr)