import math
import numpy as np
import pickle

import pandas as pd

scenario_total_balls = 240


def refine_test_data(fileName):
    df = pd.read_csv(fileName)
    df = df[(df['balls'] == scenario_total_balls)]
    return df


def baseline(df):
    scores = df[['target']].as_matrix()
    scaled_scores = scores * 0.8
    return scaled_scores


def predictor(df):
    df = df[['balls', 'wickets', 'target', 'total_overs']]

    matrix = df.as_matrix()

    pickle_in = open('gradient_boosting.pickle', 'rb')
    clf = pickle.load(pickle_in)


    predicted_scores = []
    for row in matrix:
        predicted_scores.append(clf.predict(row))

    return np.array(predicted_scores)


def duckworth_lewis(df):
    chart = {
        10: 32.1,
        9: 31.5,
        8: 30.8,
        7: 29.7,
        6: 28.3,
        5: 25.8,
        4: 22.8,
        3: 17.6,
        2: 11.4,
        1: 8.2,
        0: 0,
    }

    matrix = df[['wickets', 'target']].as_matrix()
    predicted_targets = []
    for row in matrix:
        predicted_target = row[1] * (100 - chart[10 - row[0]]) / 100
        predicted_targets.append(predicted_target)

    return np.array(predicted_targets)


def get_rmse(matrix):
    dl_errors = 0
    predicted_errors = 0

    for row in matrix:
        predicted_errors += math.pow(row[1] - row[0], 2)
        dl_errors += math.pow(row[2] - row[0], 2)
    return math.sqrt(predicted_errors / len(matrix)), math.sqrt(dl_errors / len(matrix))


df = refine_test_data('../../Test_2nd_Innings.csv')
scaled_scores = baseline(df)
predicted_scores = predictor(df)
new_df = pd.DataFrame(scaled_scores)
new_df["Predicted Scores"] = predicted_scores
new_df["DL Predicted"] = duckworth_lewis(df)
# new_df["DL Predicted"] = duckworth_lewis_ml(df)
new_df["Target"] = df['target'].as_matrix()
print get_rmse(new_df.as_matrix())
