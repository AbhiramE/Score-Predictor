import math
import numpy as np
import pickle
import csv
import pandas as pd
import sys
import duck as DL

SCENARIO_BALLLS = 240
FINAL_SCORES = {}

def populate_final_scores(fname):
    f = open(fname, 'r')

    reader = csv.reader(f)

    for row in reader:
        FINAL_SCORES[row[0]] = [row[1], row[2]]

    with open('final_score_dict.pickle', 'wb') as f:
        pickle.dump(FINAL_SCORES, f)
    pass

def refine_test_data(fileName):

    df = pd.read_csv(fileName)
    df = df[(df['balls'] == SCENARIO_BALLLS)]
    return df


def baseline(df):

    with open('final_score_dict.pickle', 'rb') as f:
        FINAL_SCORES = pickle.load(f)

    del FINAL_SCORES['match_id']
    for key in FINAL_SCORES.keys():
        df['runs'][df.match_id == int(key)] = (float(FINAL_SCORES[key][0])*float(SCENARIO_BALLLS))/float(FINAL_SCORES[key][1])
    
    scaled_scores = df[['runs']].as_matrix()
    return scaled_scores


def predictor(df):
    df = df[['balls', 'wickets', 'ground_average', 'pp_balls_left', 'total_overs']]
    #df = df[['balls', 'wickets', 'total_overs']]
    df['total_overs'] = SCENARIO_BALLLS/6
    matrix = df.as_matrix()

    #with open('../../Learning/1stInnings/random_forest.pickle', 'rb') as f:
    #with open('../../Learning/1stInnings/decision_tree.pickle', 'rb') as f:
    with open('../../Learning/1stInnings/knn.pickle', 'rb') as f:
    #with open('../../Learning/1stInnings/linearRegression.pickle', 'rb') as f:
    #with open('../../Learning/1stInnings/lasso.pickle', 'rb') as f:
    #with open('../../Learning/1stInnings/gbr.pickle', 'rb') as f:
    #with open('../../Learning/1stInnings/decision_tree_2_feat.pickle', 'rb') as f:
        clf = pickle.load(f)

    predicted_scores = []
    for row in matrix:
        predicted_scores.append(clf.predict(row))

    #print predicted_scores
    predicted_scores = np.array(predicted_scores)
    predicted_scores = np.reshape(predicted_scores, (np.shape(predicted_scores)[0], 1L))
    #print np.shape(predicted_scores)
    return predicted_scores


def duckworth_lewis(df):

    matrix = df[['runs', 'wickets']].as_matrix()
    predicted_targets = []
    for row in matrix:
        predicted_target = DL.first_innings_terminated_with(50-(SCENARIO_BALLLS/6), row[1], row[0])
        predicted_targets.append(predicted_target)

    return np.array(predicted_targets)


def get_rmse(matrix):
    dl_errors = 0
    predicted_errors = 0

    for row in matrix:
        predicted_errors += math.pow(row[1] - row[0], 2)
        dl_errors += math.pow(row[2] - row[0], 2)
    return math.sqrt(predicted_errors / len(matrix)), math.sqrt(dl_errors / len(matrix))


#Run this once to generate the dictionary of final scores
#populate_final_scores('Final_Scores_1st_Innings.csv')

df = refine_test_data('../../Learning/1stInnings/Test_1st_Innings.csv')
scaled_scores = baseline(df)
predicted_scores = predictor(df)
new_df = pd.DataFrame(scaled_scores)
new_df["ML Predicted"] = predicted_scores
#new_df = pd.DataFrame()
new_df["DL Predicted"] = duckworth_lewis(df)
print get_rmse(new_df.as_matrix())