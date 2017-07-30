from mechanize import Browser
from BeautifulSoup import BeautifulSoup
import re
import urllib
import pandas as pd
import numpy as np


def get_table_by_number(soup, number):
    tables = soup.findAll("table")
    data = []
    for row in tables[number].findAll('tr')[0:]:
        cols = row.findAll('td')
        row = []
        for col in cols:
            row.append(col)
        data.append(row)
    return data


def save_to_csv(url, pageStart, pageEnd, fileName):
    url += ';size=200'
    matrix = []
    for page_no in range(pageStart, pageEnd + 1):
        print "info from:", url + ";page=" + str(page_no)
        page = urllib.urlopen(url + ";page=" + str(page_no))
        html = page.read()
        soup = BeautifulSoup(html)
        data = get_table_by_number(soup, 2)[1:]
        for row in data:
            matrix_row = []
            for r in row:
                text = r.getText().encode("ascii")
                if len(text) != 0:
                    matrix_row.append(text)
            if len(matrix) == 0:
                matrix = np.array(matrix_row)
            else:
                matrix = np.vstack([matrix, np.array(matrix_row)])

        print matrix

        pd.DataFrame(matrix).to_csv(fileName, sep=',')


def save_to_csv_bat_and_bowl(url, pageStart, pageEnd, fileName):
    matrix = []
    for page_no in range(pageStart, pageEnd + 1):
        print "info from:", url
        page = urllib.urlopen(url)
        html = page.read()
        soup = BeautifulSoup(html)
        data = get_table_by_number(soup, 0)[1:]
        for row in data:
            matrix_row = []
            for r in row:
                text = r.getText().encode("ascii")
                name = text.split('(', 1)[0]
                matrix_row.append(name)
                if '(' in text:
                    teams = text[text.find("(") + 1:text.find(")")].split("/")
                    if len(teams) != 0:
                        matrix_row.append(teams[len(teams) - 1])
            print matrix_row
            if len(matrix) == 0:
                matrix = np.array(matrix_row)
            else:
                matrix = np.vstack([matrix, np.array(matrix_row)])

        print matrix

    pd.DataFrame(matrix).to_csv(fileName, sep=',')


def save_to_csv_teams_by_year(url, start, end, fileName):
    matrix = []
    for year in range(start, end + 1):
        print "info from:", url
        page = urllib.urlopen(url + 'id=' + str(year) + ';type=year')
        html = page.read()
        soup = BeautifulSoup(html)
        data = get_table_by_number(soup, 0)[1:]
        for row in data:
            matrix_row = []
            for r in row:
                text = r.getText().encode("ascii")
                matrix_row.append(text)

            modified_matrix_row = []
            if matrix_row[2] == matrix_row[0]:
                modified_matrix_row.append(matrix_row[0])
                modified_matrix_row.append(matrix_row[1])
                modified_matrix_row.extend(matrix_row[3:])
            elif matrix_row[2] == matrix_row[1]:
                modified_matrix_row.append(matrix_row[1])
                modified_matrix_row.append(matrix_row[0])
                modified_matrix_row.extend(matrix_row[3:])

            if len(modified_matrix_row) > 0:
                modified_matrix_row.append(matrix_row[-2].split(',')[1])
                if len(matrix) == 0:
                    matrix = np.array(modified_matrix_row)
                else:
                    matrix = np.vstack([matrix, np.array(modified_matrix_row)])

        print matrix

    pd.DataFrame(matrix).to_csv(fileName, sep=',')


def save_to_csv_batsmen_by_year(url, url2, start, end, fileName):
    matrix = []
    for year in range(start, end + 1):
        print "info from:", url + str(year) + '%2F' + str(year + 1 - 2000) + url2
        page = urllib.urlopen(url + str(year) + '%2F' + str(year + 1 - 2000) + url2)
        html = page.read()
        soup = BeautifulSoup(html)
        data = get_table_by_number(soup, 2)[1:]
        for i in range(0, 40):
            row = data[i]
            matrix_row = []
            for r in row:
                text = r.getText().encode("ascii")
                if len(text) > 0:
                    name = text.split('(', 1)[0]
                    matrix_row.append(name)
                    if '(' in text:
                        teams = text[text.find("(") + 1:text.find(")")].split("/")
                        if len(teams) != 0:
                            matrix_row.append(teams[len(teams) - 1])
            print matrix_row
            if len(matrix) == 0:
                matrix = np.array(matrix_row)
            else:
                matrix = np.vstack([matrix, np.array(matrix_row)])

        print matrix

    pd.DataFrame(matrix).to_csv(fileName, sep=',')


def save_to_csv_grounds(url, url2, pageStart, pageEnd, fileName):
    matrix = []
    for page_no in range(pageStart, pageEnd + 1):
        page = urllib.urlopen(url + str(page_no) + url2)
        html = page.read()
        soup = BeautifulSoup(html)
        data = get_table_by_number(soup, 2)[1:]
        for i in range(0, len(data)-1, 2):
            row = data[i]
            row2 = data[i+1]
            print row2
            matrix_row = [row[0].getText().encode("ascii"),
                          int(row2[6].getText().encode("ascii"))/ (int(row2[2].getText().encode("ascii"))*2)]
            if len(matrix) == 0:
                matrix = np.array(matrix_row)
            else:
                matrix = np.vstack([matrix, np.array(matrix_row)])

        print matrix

        pd.DataFrame(matrix).to_csv(fileName, sep=',')


# url = 'http://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;filter=advanced;orderby=team_score;size=200
# ;spanmin1=05+Jan+1994;spanval1=span;template=results;type=team;view=innings' saveToCsv(url, 1, 30,
# "Match_Details.csv")


# url = 'http://stats.espncricinfo.com/ci/content/records/83548.html'
# save_to_csv_bat_and_bowl(url, 1, 1, "Batsmen/MostRuns.csv")

# url = 'http://stats.espncricinfo.com/ci/content/records/93276.html'
# save_to_csv_bat_and_bowl(url, 1, 1, "Bowlers/MostWickets.csv")

# ,Winner,Loser,Margin,Ground,Date_of_game,ODI_code,Year
# url = 'http://stats.espncricinfo.com/ci/engine/records/team/match_results.html?class=2;'
# save_to_csv_teams_by_year(url, 2007, 2017, "team_records_years_wise.csv")

# url = 'http://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;filter=advanced;orderby=runs;season='
# url2 = ';size=200;template=results;type=batting;view=season'
# save_to_csv_batsmen_by_year(url, url2, 2007, 2016, "Batsmen/bowlers_records_years_wise.csv")

# url = 'http://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;filter=advanced;orderby=wickets;season='
# url2 = ';size=200;template=results;type=bowling;view=season'
# save_to_csv_bat_and_bowl_by_year(url, url2, 2007, 2016, "Bowlers/bowlers_records_years_wise.csv")

# ,match_id,balls,score,wickets,ground,ground_average,wickets_remaining,balls_remaining
url = 'http://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;page='
url2 = ';spanmin1=13+Jun+2006;spanval1=span;template=results;type=aggregate;view=ground'
save_to_csv_grounds(url, url2, 1, 2, "Ground_Stats.csv")
