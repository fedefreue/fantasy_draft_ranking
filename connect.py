from yahoo_oauth import OAuth2
import pandas as pd

#https://vicmora.github.io/blog/2017/03/17/yahoo-fantasy-sports-api-authentication
conn = OAuth2(None, None, from_file='private.json')
if not conn.token_is_valid():
    conn.refresh_access_token()

url = 'https://fantasysports.yahooapis.com/fantasy/v2/leagues;league_keys=nfl.l.254924'
r = conn.session.get(url, params={'format': 'json'})

url = 'https://fantasysports.yahooapis.com/fantasy/v2/league/nfl.l.254924/players;status=A'
r = conn.session.get(url, params={'format': 'xml'})

print(r.json())

data = pd.json_normalize(r.json())
print(data)


#lg = fya.League(conn,'254924')

#draft_res = lg.teams()
#print(draft_res)


#oauth.refresh_access_token()
#response = oauth.session.get(url, params=payload)


"""
if not oauth.token_is_valid():
    oauth.refresh_access_token()
# Example
response = oauth.session.get(url, params=payload)
"""


"""
OATH_APP_ID = 'MWsTtRHp'

REDIRECT_URI = 'localhost:8080'
"""