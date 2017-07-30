import math
import numpy as np
import pickle

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
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

    with open('adaboost.pickle', 'wb') as f:
        pickle.dump(clf, f)

    return math.sqrt(mean_squared_error(y_pred=y_predict, y_true=testY))


def pipeline(X_tr, Y_tr):
    X_train, X_validate, y_train, y_validate = train_test_split(X_tr, Y_tr, test_size=0.2)

    '''
    # Get the best hyperparameters on selected features
    param_grid = {
                  "base_estimator__splitter": ["best", "random"],
                  "n_estimators": [2000, 3000]
                  }

    decisionTree = DecisionTreeRegressor(random_state=11, max_features="auto", max_depth=None)
    method = AdaBoostRegressor(base_estimator=decisionTree)
    clf = hyperparameter_selection(X_train, y_train, param_grid, method)
    print "Pipeline Best Hyperparameter ", clf.best_params_["base_estimator__splitter"], \
        clf.best_params_["n_estimators"]
    '''
    # Run the regressor on the selected features and best hyperparameter
    decisionTree = DecisionTreeRegressor(random_state=11, max_features="auto", splitter='best')
    adaBoost = AdaBoostRegressor(n_estimators=2000, base_estimator=decisionTree)
    rmse = regress(X_train, y_train, X_validate, y_validate, adaBoost)
    print "Pipeline RMSE ", rmse


X_tr, y_tr = get_features_and_labels('../../Train_2nd_Innings.csv')
pipeline(X_tr, y_tr)
