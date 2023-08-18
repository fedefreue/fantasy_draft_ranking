import sqlite3
import pandas as pd
import scrape


def initialize(dbConnection, script: str):
    cur = dbConnection.cursor()
    cur.executescript(script)
    cur.commit()
    cur.close()


def data_rawdl(data_years, dbConnection):
    # Append the 1 - the lowest value in data_years to data_years
    data_years.insert(0, min(data_years) - 1)

    dbConnection.execute("DROP TABLE IF EXISTS rawData")
    dbConnection.commit()

    for year in data_years:
        print("Loading " + str(year) + "...")
        raw_data = scrape.scrape_data(year)
        # rawData = pd.read_csv(f'https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/{year}.csv')
        # Add a column to rawData with the year
        raw_data["seasonYear"] = year
        raw_data.to_sql("rawData", dbConnection, if_exists="append", index=False)


def data_gen_year_list(start_year, num_years):
    year_list = []
    for i in range(num_years):
        year_list.append(start_year - i)
    return year_list


def data_format(dbConnection):
    dbConnection.execute(
        """UPDATE rawData
    SET Player = REPLACE(Player, '+', '')
    WHERE Player LIKE '%+';"""
    )

    dbConnection.execute(
        """UPDATE rawData
    SET Player = REPLACE(Player, '*', '')
    WHERE Player LIKE '%*';"""
    )

    # Remove incorrect rows
    dbConnection.execute(
        """DELETE FROM rawData 
    WHERE Pos IS NULL 
    OR Pos = '0'; """
    )

    dbConnection.execute(
        """
INSERT INTO positions (position)
 SELECT DISTINCT Pos
 FROM rawData;"""
    )

    dbConnection.commit()

    # Load players & rosters
    dbConnection.execute(
        """INSERT INTO players (name, position)
    SELECT Player, positions.id
    FROM rawData
    JOIN positions
    ON rawData.Pos = positions.position
    GROUP BY Player, positions.id;"""
    )

    dbConnection.execute(
        """INSERT INTO rosters (player_id, season_year, tm)
    SELECT players.id, rawData.seasonYear, rawData.Tm 
    FROM rawData
    JOIN players
    ON rawData.Player = players.name
    GROUP BY players.id, rawData.Tm, rawData.seasonYear;"""
    )

    dbConnection.commit()

    dbConnection.execute(
        """WITH cte AS ( 
    SELECT p.id
        , CAST(seasonYear AS INTEGER) AS season_year
        , Tm AS team
        , Pos AS pos
        , CAST(Age AS INTEGER) as age
        , CAST(G AS INTEGER) as games_played
        , CAST(GS AS INTEGER) as games_started
        , CAST(Tgt AS INTEGER) as targets
        , CAST(Rec AS INTEGER) as receptions
        , CAST(PassingYds AS INTEGER) as passing_yards
        , CAST(PassingTD AS INTEGER) as passing_td
        , CAST(PassingAtt AS INTEGER) as passing_att
        , CAST(RushingYds AS INTEGER) as rushing_yards
        , CAST(RushingTD AS INTEGER) as rushing_td
        , CAST(RushingAtt AS INTEGER) as rushing_att
        , CAST(ReceivingYds AS INTEGER) as receiving_yards
        , CAST(ReceivingTD AS INTEGER) as receiving_td
        , CAST(Int AS INTEGER) as interceptions
        , CAST(Fumbles AS INTEGER) as fumbles
        , CAST(FumblesLost AS INTEGER) as fumbles_lost
    FROM rawData rd
    JOIN players p
        ON rd.Player = p.name )
 INSERT INTO features
 SELECT a.*
    , b.season_year
    , CAST(IIF(a.team = b.team, 0, 1) AS INTEGER) AS traded
    , b.receptions * c.receptions
        + b.passing_yards * c.passing_yards
        + b.passing_td * c.passing_td
        + b.rushing_yards * c.rushing_yards
        + b.rushing_td * c.rushing_td
        + b.receiving_yards * c.receiving_yards
        + b.receiving_td * c.receiving_td
        + b.interceptions * c.interceptions
        + b.fumbles * c.fumbles
 FROM cte a
 JOIN cte b
    ON a.id = b.id
    AND (a.season_year + 1) = b.season_year
 JOIN points_slate c;"""
    )

    dbConnection.commit()

    dbConnection.execute(
        """ALTER TABLE features
 ADD COLUMN season_rank INTEGER;"""
    )

    dbConnection.execute(
        """UPDATE features
    SET season_rank = RANK() OVER (ORDER BY a.points DESC)
    FROM (SELECT * FROM features) AS a
    WHERE a.id = features.id AND a.season_year_features = features.season_year_features;"""
    )

    dbConnection.commit()

    # Return a pd.DataFrame with features
    return pd.read_sql_query("SELECT * FROM features", dbConnection)
