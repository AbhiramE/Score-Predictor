import pandas as pd
import yaml
import numpy as np
import os


def get_venue_average(venue):
    df_grounds = pd.read_csv("/home/abhis3798/codebase/ML/DL Score Predictor/Project/Collection/Ground_Stats.csv")
    grounds = df_grounds.as_matrix()
    for g in grounds:
        if venue in g[1]:
            return g[-1]
    else:
        return 250


def get_data():
    matrix_1_test = []
    matrix_2_test = []
    matrix_1_train = []
    matrix_2_train = []
    new_matrix= []
    i = 0
    total_overs = 50
    for filename in os.listdir('/home/abhis3798/codebase/ML/odis'):
        matrix_1 = []
        matrix_2 = []
        i += 1
        print i
        with open('/home/abhis3798/codebase/ML/odis/' + filename, 'r') as f:
            doc = yaml.load(f)

            if doc["info"]["gender"] != "male":
                continue

            data = doc["innings"][0]['1st innings']['deliveries']
            data2 = None
            if len(doc["innings"]) > 1:
                data2 = doc["innings"][1]['2nd innings']['deliveries']

            venue = doc["info"]["venue"]
            runs = 0
            wicket = 0
            balls = 0
            for datum in data:
                for key, value in datum.items():
                    runs += int(value['runs']['total'])
                    if 'extras' not in value:
                        balls += 1
                    elif 'legbyes' in value['extras'] \
                            or 'byes' in value['extras']:
                        balls += 1

                    if 'wicket' in value:
                        wicket += 1

                    venue_ave = get_venue_average(venue)
                    powerplay_balls = 0
                    if balls < 60:
                        powerplay_balls = balls - 60
                    row = [i, balls, runs, wicket, venue_ave, powerplay_balls, total_overs]
                    if len(matrix_1) == 0:
                        matrix_1 = np.array(row)
                    else:
                        matrix_1 = np.vstack([matrix_1, np.array(row)])
            target = runs
            if balls > 295 or wicket == 10:
                choices = ['test', 'train']
                choice = np.random.choice(choices, p=[0.2, 0.8])
                if choice == 'test':
                    if len(matrix_1_test) == 0:
                        matrix_1_test = matrix_1.copy()
                    else:
                        matrix_1_test = np.vstack([matrix_1_test, matrix_1])
                else:
                    if len(matrix_1_train) == 0:
                        matrix_1_train = matrix_1.copy()
                    else:
                        matrix_1_train = np.vstack([matrix_1_train, matrix_1])

        if len(new_matrix) == 0:
            new_matrix = np.array([i, target, balls])
        else:
            new_matrix = np.vstack([new_matrix, np.array([i, target, balls])])

            '''
            runs = 0
            wicket = 0
            balls = 0
            if data2 is not None:
                for datum in data2:
                    for key, value in datum.items():
                        runs += int(value['runs']['total'])
                        if 'extras' not in value:
                            balls += 1
                        elif 'legbyes' in value['extras'] \
                                or 'byes' in value['extras']:
                            balls += 1
                        if 'wicket' in value:
                            wicket += 1
                        venue_ave = get_venue_average(venue)
                        powerplay_balls = 0
                        if balls < 60:
                            powerplay_balls = balls - 60
                        row = [i, balls, runs, wicket, venue_ave, powerplay_balls, target, total_overs]
                        if len(matrix_2) == 0:
                            matrix_2 = np.array(row)
                        else:
                            matrix_2 = np.vstack([matrix_2, np.array(row)])
            if runs >= target or wicket == 10 or balls >= 295:
                choices = ['test', 'train']
                choice = np.random.choice(choices, p=[0.2, 0.8])
                if choice == 'test':
                    if len(matrix_2_test) == 0:
                        matrix_2_test = matrix_2.copy()
                    else:
                        matrix_2_test = np.vstack([matrix_2_test, matrix_2])
                else:
                    if len(matrix_2_train) == 0:
                        matrix_2_train = matrix_2.copy()
                    else:
                        matrix_2_train = np.vstack([matrix_2_train, matrix_2])
            '''
    pd.DataFrame(new_matrix).to_csv("Final_Scores_1st_Innings.csv",sep=',')
    # pd.DataFrame(matrix_1_train).to_csv("Train_1st_Innings.csv", sep=',')
    # pd.DataFrame(matrix_2_train).to_csv("Train_2nd_Innings.csv", sep=',')
    # pd.DataFrame(matrix_1_test).to_csv("Test_1st_Innings.csv", sep=',')
    # pd.DataFrame(matrix_2_test).to_csv("Test_2nd_Innings.csv", sep=',')


def modify_data(fileName):
    df = pd.read_csv(fileName)
    rows = df.as_matrix()
    matrix = []

    for row in rows:
        new_row = np.array(row.copy())
        balls = int(new_row[2])
        non_powerplay_balls = int(new_row[-1])

        if balls < 60:
            poweplay_balls = 60 - balls
            non_powerplay_balls -= poweplay_balls
        else:
            poweplay_balls = 0
        new_row = np.append(new_row[1:-1], [poweplay_balls, non_powerplay_balls])
        if len(matrix) == 0:
            matrix = np.array(new_row)
        else:
            matrix = np.vstack([matrix, new_row])
    pd.DataFrame(matrix).to_csv("second_innings.csv", sep=',')


# ,match_id,balls,runs,wickets,ground_average,pp_balls_left,total_overs
get_data()
