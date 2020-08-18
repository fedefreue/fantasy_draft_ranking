import pandas as pd
import yahoo_fantasy_api as yfa

def pullStatsAndPoints(season: int, league: yfa.League, points: dict, position: str, players: list):
    
    priorSeason = int(season) - 1
    thisSeasonStats = pd.DataFrame(league.player_stats(players, 'season', season = season))

    # Calculate offensive points
    calculateOffensivePoints(thisSeasonStats, points) 

    thisSeasonPoints = thisSeasonStats[['player_id','Points']].copy()
    
    lastSeasonStats = pd.DataFrame(league.player_stats(players, 'season', season = priorSeason))
    df = thisSeasonPoints.merge(lastSeasonStats, on = 'player_id')
    df = df.drop(columns = ['name'])    

    return df

def calculateOffensivePoints(thisSeasonStats, points):
    thisSeasonStats['Points'] = thisSeasonStats['Pass Yds'] * points['Pass Yds'] + thisSeasonStats['Pass TD'] * points['Pass TD'] + thisSeasonStats['Int'] * points['Int'] + thisSeasonStats['Rush Att'] * points['Rush Att'] + thisSeasonStats['Rush TD'] * points['Rush TD'] + thisSeasonStats['Rec'] * points['Rec'] + thisSeasonStats['Rec Yds'] * points['Rec Yds'] + thisSeasonStats['Rec TD'] * points['Rec TD'] + thisSeasonStats['Ret TD'] * points['Ret TD'] + thisSeasonStats['2-PT'] * points['2-PT'] + thisSeasonStats['Fum Lost'] * points['Fum Lost'] + thisSeasonStats['Fum Ret TD'] * points['Fum Ret TD'] 

def generateTrainingTable(seasonList: list, league: yfa.League, points: dict, positions: list):
    masterTable = pd.DataFrame()
    
    for s in seasonList:
        for p in positions:
            freeAgents = pd.DataFrame(league.free_agents(p))
            freeAgentList = list(freeAgents['player_id'])
            
            interimTable = pullStatsAndPoints(s, league, points, p, freeAgentList)
            masterTable = masterTable.append(interimTable)

    # Remove 0 rows
    masterTable = masterTable.loc[masterTable['Points'] != 0]

    return masterTable