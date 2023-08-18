-- SQL
CREATE TABLE IF NOT EXISTS dbInitialize (
    bool bit NOT NULL,
    timestamp date NOT NULL );

INSERT INTO dbInitialize (bool, timestamp) VALUES (1, DATE('now'));

DROP TABLE IF EXISTS players;
DROP TABLE IF EXISTS rawData;
DROP TABLE IF EXISTS positions;
DROP TABLE IF EXISTS teams;
DROP TABLE IF EXISTS rosters;
DROP TABLE IF EXISTS features;
DROP TABLE IF EXISTS points_slate;

CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY
    , name TEXT NOT NULL
    , position int );

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY
    , position TEXT NOT NULL );

CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY
    , teamName TEXT NOT NULL
    , teamAbbrev TEXT NOT NULL );

CREATE TABLE IF NOT EXISTS rosters (
    player_id INTEGER
    , season_year INTEGER
    , tm TEXT );

CREATE TABLE IF NOT EXISTS features (
    id INTEGER
    , season_year_features INTEGER
    , team TEXT
    , pos TEXT
    , age INTEGER
    , games_played INTEGER
    , games_started INTEGER
    , targets INTEGER
    , receptions INTEGER
    , passing_yards INTEGER
    , passing_td INTEGER
    , passing_att INTEGER
    , rushing_yards INTEGER
    , rushing_td INTEGER
    , rushing_att INTEGER
    , receiving_yards INTEGER
    , receiving_td INTEGER
    , interceptions INTEGER
    , fumbles INTEGER
    , fumbles_lost INTEGER
    , season_year_points INTEGER
    , traded INTEGER
    , points FLOAT );

CREATE TABLE IF NOT EXISTS points_slate (
    receptions INTEGER
    , passing_yards INTEGER
    , passing_td INTEGER
    , rushing_yards INTEGER
    , rushing_td INTEGER
    , receiving_yards INTEGER
    , receiving_td INTEGER
    , interceptions INTEGER
    , fumbles INTEGER);

INSERT INTO points_slate VALUES (
    1,
    0.04,
    4,
    0.1,
    6,
    0.1,
    6,
    -2,
    -2 );
