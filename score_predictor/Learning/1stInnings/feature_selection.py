import math
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd


def get_features_and_labels(fileName):
    df = pd.read_csv(fileName)
    df = df[['balls', 'wickets', 'ground_average', 'pp_balls_left', 'total_overs', 'runs']]
    df_features = df.drop('runs', axis=1)
    df_label = df[['runs']]
    return df_features.as_matrix(), df_label.as_matrix()


def select_features(trainX, trainY, testX, testY):
    clf = Lasso()
    selector = SelectKBest(k=5)
    selector.fit(trainX, trainY)
    trainX = selector.fit_transform(trainX, trainY)
    testX = selector.transform(testX)
    print "Selected features", selector.get_support()

    clf.fit(trainX, trainY)
    y_predict = clf.predict(testX)
    print "Feature Selection RMSE ", 4, math.sqrt(mean_squared_error(y_pred=y_predict, y_true=testY))


X_tr, y_tr = get_features_and_labels('../Train_1st_Innings.csv')
X_train, X_validate, y_train, y_validate = train_test_split(X_tr, y_tr, test_size=0.2)
select_features(X_train, y_train, X_validate, y_validate)
