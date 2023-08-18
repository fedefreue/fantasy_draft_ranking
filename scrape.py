# Adapted from https://github.com/bendominguez0111/fantasy-csv-data/

import requests
from bs4 import BeautifulSoup
import pandas as pd

pd.set_option("display.max_columns", None)


def scrape_data(year: int):
    url = "https://www.pro-football-reference.com/years/{}/fantasy.htm".format(year)

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", {"id": "fantasy"})
    df = pd.read_html(str(table))[0]
    df.columns = df.columns.droplevel(level=0)

    # Remove filler rows
    df = df.loc[df["Player"] != "Player"]

    # Fix player names that have astericks
    df["Player"] = df["Player"].apply(lambda x: x.split("*")[0].strip())

    # Column names:
    # Player,Tm,Pos,Age,G,GS,Tgt,Rec,PassingYds,PassingTD,PassingAtt,RushingYds,
    # RushingTD,RushingAtt,ReceivingYds,ReceivingTD,FantasyPoints,Int,Fumbles,FumblesLost

    df["PassingYds"] = df["Yds"].iloc[:, 0]
    df["RushingYds"] = df["Yds"].iloc[:, 1]
    df["ReceivingYds"] = df["Yds"].iloc[:, 2]

    df["PassingTD"] = df["TD"].iloc[:, 0]
    df["RushingTD"] = df["TD"].iloc[:, 1]
    df["ReceivingTD"] = df["TD"].iloc[:, 2]

    df["PassingAtt"] = df["Att"].iloc[:, 0]
    df["RushingAtt"] = df["Att"].iloc[:, 1]

    df = df.rename(
        columns={
            "FantPos": "Pos",
            "FantPt": "FantasyPoints",
            "Fmb": "Fumbles",
            "FL": "FumblesLost",
        }
    )

    df = df[
        [
            "Player",
            "Tm",
            "Pos",
            "Age",
            "G",
            "GS",
            "Tgt",
            "Rec",
            "PassingYds",
            "PassingTD",
            "PassingAtt",
            "RushingYds",
            "RushingTD",
            "RushingAtt",
            "ReceivingYds",
            "ReceivingTD",
            "FantasyPoints",
            "Int",
            "Fumbles",
            "FumblesLost",
        ]
    ]

    # df.to_csv('yearly/2020.csv', index=False)
    return df
