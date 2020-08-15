import logging
import os
import pprint
import unittest
import warnings
from unittest import skip, TestCase

from yfpy import Data
from yfpy.models import Game, StatCategories, User, Scoreboard, Settings, Standings, League, Player, Team, \
    TeamPoints, TeamStandings, Roster
from yfpy.query import YahooFantasySportsQuery






OATH_APP_ID = 'MWsTtRHp'
#CLIENT_ID = 'dj0yJmk9SW9mNVdqRndRdnp5JmQ9WVdrOVRWZHpWSFJTU0hBbWNHbzlNQT09JnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PTE1'
#CLIENT_SECRET = 'efc11770248263c8a689f7d9ddf0c1a2cddf73b3'


REDIRECT_URI = 'localhost:8080'

AUTHORIZE_URL = 'https://api.login.yahoo.com/oauth/v2/request_auth'
ACCESS_TOKEN_URL = 'https://api.login.yahoo.com/oauth/v2/get_token'
request_token_url = 'https://api.login.yahoo.com/oauth/v2/get_request_token'
