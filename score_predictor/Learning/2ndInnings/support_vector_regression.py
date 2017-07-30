import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import pickle
import sys

orig_stdout = sys.stdout
f_redirect = open('out_svr.txt', 'w')
sys.stdout = f_redirect


def get_features_and_labels(fileName):
    df = pd.read_csv(fileName)
    df = df[['balls', 'wickets', 'ground_average', 'pp_balls_left', 'total_overs', 'runs']]
    df_features = df.drop('runs', axis=1)
    df_label = df[['runs']]
    return df_features.as_matrix(), df_label.as_matrix()


# Get the best hyper-parameters using GridSearchCV
def hyperparameter_selection(train_x, train_y, param_grid, method):
    train_y = train_y.reshape(len(train_y), )
    clf = GridSearchCV(method, param_grid)
    clf.fit(train_x, train_y)
    return clf


# Regress on Lasso to predict the Classes for the test set
def regress(trainX, trainY, testX, testY, clf):
    clf.fit(trainX, trainY)
    y_predict = clf.predict(testX)

    with open('lasso.pickle', 'wb') as f:
        pickle.dump(clf, f)

    return math.sqrt(mean_squared_error(y_pred=y_predict, y_true=testY))


def pipeline(X_tr, Y_tr):
    X_train, X_validate, y_train, y_validate = train_test_split(X_tr, Y_tr, test_size=0.2)

    # Get the best hyperparameters on selected features
    parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [0.25, 0.5, 1, 2, 4, 8],
                  'gamma': [1e-7, 1e-4, 1e-2, 1]}
    svr = SVR()
    clf = hyperparameter_selection(X_train, y_train, parameters, svr)
    print "Pipeline Best Hyperparameter ", clf.best_params_['C']
    c = clf.best_params_['C']
    gamma = clf.best_params_['gamma']
    kernel = clf.best_params_['kernel']

    # Run the regressor on the selected features and best hyperparameter
    svr = SVR(C=c, gamma=gamma, kernel=kernel)
    rmse = regress(X_train, y_train, X_validate, y_validate, svr)
    print "Pipeline RMSE ", rmse


X_tr, y_tr = get_features_and_labels('../Train_1st_Innings.csv')
pipeline(X_tr, y_tr)
sys.stdout = orig_stdout
f_redirect.close()
