import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2

import playerData
import rankPlayers

def connect(fileName:str):
    conn = OAuth2(None, None, from_file=fileName)

    if not conn.token_is_valid():
        conn.refresh_access_token()
    
    return conn

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