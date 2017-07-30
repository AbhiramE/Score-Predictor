import math
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle

import pandas as pd

scenario_total_balls = 240


def get_features_and_labels(fileName):
    df = pd.read_csv(fileName)
    df = df[['balls', 'wickets', 'target', 'total_overs', 'runs']]
    df_features = df.drop('runs', axis=1)
    df_label = df[['runs']]
    return df_features.as_matrix(), df_label.as_matrix()


# Regress on GBR to predict the Classes for the test set
def regress(trainX, trainY, testX, testY, clf):
    trainY = np.ravel(trainY)
    clf.fit(trainX, trainY)
    y_predict = clf.predict(testX)

    with open('gradient_boosting.pickle', 'wb') as f:
        pickle.dump(clf, f)

    return math.sqrt(mean_squared_error(y_pred=y_predict, y_true=testY))


def pipeline(X_tr, Y_tr):
    X_train, X_validate, y_train, y_validate = train_test_split(X_tr, Y_tr, test_size=0.2)

    # Run the regressor on the selected features and best hyperparameter
    decisionTree = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01,
                                             max_depth=6)
    rmse = regress(X_train, y_train, X_validate, y_validate, decisionTree)
    print "Pipeline RMSE ", rmse


X_tr, y_tr = get_features_and_labels('../../Train_2nd_Innings.csv')
pipeline(X_tr, y_tr)
