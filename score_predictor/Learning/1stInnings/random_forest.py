import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import pickle
import sys
from sklearn import tree
from sklearn import ensemble

orig_stdout = sys.stdout
f_redirect = open('out_random_forest.txt', 'w')
sys.stdout = f_redirect


def get_features_and_labels(fileName):
    df = pd.read_csv(fileName)
    df = df[['balls', 'wickets', 'ground_average', 'pp_balls_left', 'total_overs', 'runs']]
    df_features = df.drop('runs', axis=1)
    df_label = df[['runs']]
    return df_features.as_matrix(), df_label.as_matrix()


def dump_train_pickle(X_tr, y_tr):
    with open('1st_innings_train_Xtr.pickle', 'wb') as f:
        pickle.dump(X_tr, f)

    with open('1st_innings_train_ytr.pickle', 'wb') as f:
        pickle.dump(y_tr, f)

def read_features_labels_pickle():

    with open('1st_innings_train_Xtr.pickle', 'rb') as f:
        X_tr = pickle.load(f)

    with open('1st_innings_train_ytr.pickle', 'rb') as f:
        y_tr = pickle.load(f)    

    return (X_tr, y_tr)

# Get the best hyper-parameters using GridSearchCV
def hyperparameter_selection(train_x, train_y, param_grid=None, method=None):

    print "In hyperparameter selection now"
    train_y = train_y.reshape(len(train_y), )
    if(param_grid != None):
        clf = GridSearchCV(method, param_grid)
    else:
        clf = method

    clf.fit(train_x, train_y)
    return clf


# Regress on Lasso to predict the Classes for the test set
def regress(trainX, trainY, testX, testY, clf):
        
    print "In regresssion now"
    y_predict = clf.predict(testX)
    with open('random_forest.pickle', 'wb') as f:
        pickle.dump(clf, f)

    return math.sqrt(mean_squared_error(y_pred=y_predict, y_true=testY))


def pipeline(X_tr, Y_tr):

    print "In pipeline now"
    X_train, X_validate, y_train, y_validate = train_test_split(X_tr, Y_tr, test_size=0.2)
    # Run the regressor on the selected features and best hyperparameter
    param_grid = {"n_estimators": [1000, 2000, 3000], "max_features": ["sqrt"], "min_samples_split":[2, 4, 6]}
    clf = hyperparameter_selection(X_train, y_train, param_grid=param_grid, method=ensemble.RandomForestRegressor(random_state=44))
    rmse = regress(X_train, y_train, X_validate, y_validate, clf)
    print "Pipeline RMSE ", rmse


# X_tr, y_tr = get_features_and_labels('Train_1st_Innings.csv')
# dump_train_pickle(X_tr, y_tr)
X_tr, y_tr = read_features_labels_pickle()
pipeline(X_tr, y_tr)
sys.stdout = orig_stdout
f_redirect.close()