from yahoo_oauth import OAuth2
import pandas as pd
import yahoo_fantasy_api as yfa
import playerData

#https://vicmora.github.io/blog/2017/03/17/yahoo-fantasy-sports-api-authentication

conn = OAuth2(None, None, from_file='private.json')
if not conn.token_is_valid():
    conn.refresh_access_token()

league = yfa.League(conn,'nfl.l.254924')

#freeAgents = pd.DataFrame(league.free_agents(position))
#freeAgentList = list(freeAgents['player_id'])
seasonList = [2016, 2017, 2018, 2019]
positions = ['RB', 'QB', 'WR', 'TE', 'K']
points = {
    'Pass Yds': 0.04,
    'Pass TD': 4,
    'Int': -1,
    'Rush Att': 0.1,
    'Rush TD': 6,
    'Ret TD': 6,
    'Rec': 1,
    'Rec Yds': 0.1,
    'Rec TD': 6,
    '2-PT': 2,
    'Fum Lost': -2,
    'Fum Ret TD': 2
}

playerData.generateTrainingTable(seasonList, league, points, positions)
