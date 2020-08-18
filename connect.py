import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2

import playerData
import rankPlayers

#https://vicmora.github.io/blog/2017/03/17/yahoo-fantasy-sports-api-authentication
#conn.session.get('https://fantasysports.yahooapis.com/fantasy/v2/league/XXX.l.25494')

def connect(fileName:str):
    conn = OAuth2(None, None, from_file=fileName)

    if not conn.token_is_valid():
        conn.refresh_access_token()
    
    return conn

conn = connect('private.json')

#league = yfa.League(conn,'nfl.l.254924')

#freeAgents = pd.DataFrame(league.free_agents(position))
#freeAgentList = list(freeAgents['player_id'])
seasonList = [2018, 2019]
positions = ['RB', 'QB', 'WR', 'TE']
points = {
    'Pass Yds':     0.04,
    'Pass TD':      4,
    'Int':          -1,
    'Rush Att':     0.1,
    'Rush TD':      6,
    'Ret TD':       6,
    'Rec':          1,
    'Rec Yds':      0.1,
    'Rec TD':       6,
    '2-PT':         2,
    'Fum Lost':     -2,
    'Fum Ret TD':   2
}

"""
masterTable = playerData.generateTrainingTable(seasonList, league, points, positions)

masterTable

features = masterTable.drop(columns = ['position_type', 'player_id', 'Points'])
y_train = masterTable['Points']

features = features.to_numpy()
y_train = y_train.to_numpy()

reg = sklearn.linear_model.LinearRegression().fit(features, y_train)

print(reg.score(features, y_train))
plt.scatter(reg.predict(features), y_train )
plt.show()
"""
"""
staging = rankPlayers.rankPlayers('QB', league, reg, 2019)
staging.to_csv('test_file.csv')
"""