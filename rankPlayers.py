import pandas as pd
import yahoo_fantasy_api as yfa
import sklearn.linear_model


def rankPlayers(
    position: str,
    league: yfa.League,
    model: sklearn.linear_model.LinearRegression,
    season: int,
):
    players = pd.DataFrame(league.free_agents(position))
    playerList = list(players["player_id"])

    thisSeasonStats = pd.DataFrame(
        league.player_stats(playerList, "season", season=season)
    )
    thisSeasonStats_train = thisSeasonStats.drop(
        columns=["player_id", "position_type", "name"]
    )

    points = model.predict(thisSeasonStats_train)
    thisSeasonStats["Points"] = points
    returnTable = thisSeasonStats[["player_id", "name", "Points"]].sort_values(
        by=["Points"], ascending=False
    )

    return returnTable
