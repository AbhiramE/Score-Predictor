import pandas as pd
import numpy as np


def clean_innings_for_scenario_1():
    df = pd.read_csv("Data2.csv")
    df_grounds = pd.read_csv("Ground_Stats.csv")
    data = df.as_matrix()
    grounds = df_grounds.as_matrix()
    matrix = []
    data = data[data[:, 2] == 215]
    count = 0
    for row in data:
        matrix_row = []
        for g in grounds:
            if row[-1] in g[1].encode("ascii"):
                matrix_row = row.copy()
                matrix_row =np.append(matrix_row[1:], [g[-1], 10-int(row[-2]),300-215])
                break
        if len(matrix_row) == 0:
            matrix_row = row.copy()
            matrix_row = np.append(matrix_row[1:], [250, 10-int(row[-2]),300-215])
        if len(matrix) == 0:
            matrix = np.array(matrix_row)
        else:
            matrix = np.vstack([matrix, np.array(matrix_row)])

    pd.DataFrame(matrix).to_csv("second_innings.csv", sep=',')


clean_innings_for_scenario_1()
