import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import pickle


def get_features_and_labels(fileName):
    df = pd.read_csv(fileName)
    df = df[['balls', 'wickets', 'ground_average', 'pp_balls_left', 'total_overs', 'runs']]
    df_features = df.drop('runs', axis=1)
    df_label = df[['runs']]
    return df_features.as_matrix(), df_label.as_matrix()

def read_features_labels_pickle():

    with open('1st_innings_train_Xtr.pickle', 'rb') as f:
        X_tr = pickle.load(f)

    with open('1st_innings_train_ytr.pickle', 'rb') as f:
        y_tr = pickle.load(f)    

    return (X_tr, y_tr)
    
def linear_regression(X_tr, y_tr):
    X_train, X_validate, y_train, y_validate = train_test_split(X_tr, y_tr, test_size=0.2)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X=X_validate)
    print math.sqrt(mean_squared_error(y_pred=y_predicted, y_true=y_validate))

    with open('linearRegression.pickle', 'wb') as f:
        pickle.dump(clf, f)


X_tr, y_tr = read_features_labels_pickle()
linear_regression(X_tr, y_tr)
