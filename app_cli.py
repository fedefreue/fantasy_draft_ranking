import data_prep
import model
import sqlite3
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import tensorflow as tf

layer1 = 15
layer2 = 7
current_year = 2022
years_to_train = 15
positions = ["RB", "QB", "WR", "TE"]
raw_features_name = "rawData"
raw_apply_name = "rawData_apply"

dbConnection = sqlite3.connect("db.sqlite3")

init_bit = input("Initialize DB? (y/n)")
if init_bit == "y" or init_bit == "yes":
    with open("schema.sql", "r") as sql_file:
        sql_script = sql_file.read()
        cur = dbConnection.cursor()
        cur.executescript(sql_script)
        # cur.commit()
        cur.close()

train_bit = input("Train model set? (y/n)")
if train_bit == "y" or train_bit == "yes":
    # Check if the initializated bit exists
    if (
        dbConnection.execute("SELECT COUNT(*) FROM dbInitialize WHERE bool = 1").fetchall()
    ) is not None:
        print("Database has been initialized")
    else:
        raise Exception("Database has not been initialized")

    # For each value in dataYears, scrape the data into a raw_data table
    # dbConnection.execute("DROP TABLE IF EXISTS rawData")
    # dbConnection.commit()
    data_years = data_prep.gen_year_list(current_year, years_to_train)
    data_prep.raw_dl(data_years, dbConnection, raw_features_name)
    data_table = data_prep.format(dbConnection, raw_features_name, for_training=1)

    model.optimize_set(
        db_connection=dbConnection,
        layer1=layer1,
        layer2=layer2,
        staging_table_name="features",
        save_model=1,
        file_name="debug",
        by_position=1,
        positions=["RB", "QB", "WR", "TE"],
        verbose=1,
    )

apply_bit = input("Apply saved models? (y/n)")
if apply_bit == "y" or apply_bit == "yes":
    data_prep.raw_dl([current_year], dbConnection, raw_apply_name)
    data_table = data_prep.format(dbConnection, raw_apply_name, for_training=0)

    debug_model_all = tf.keras.models.load_model('debug_all.tf')

    for position in positions:
        print('Applying ' + position + ' model...')
        debug_model_position = tf.keras.models.load_model('debug_' + position + '.tf')
        
        apply_set = pd.read_sql_query("SELECT a.*, b.name FROM features a JOIN players b ON a.id = b.id WHERE a.season_year_features = 2022 AND a.pos = '" + position + "'", dbConnection)

        apply_set_features = pd.DataFrame(
                apply_set,
                columns=[
                    "age",
                    "games_played",
                    "games_started",
                    "targets",
                    "receptions",
                    "passing_yards",
                    "passing_td",
                    "passing_att",
                    "rushing_yards",
                    "rushing_td",
                    "rushing_att",
                    "receiving_yards",
                    "receiving_td",
                    "interceptions",
                    "fumbles",
                    "fumbles_lost",
                ],
            )

        apply_set['pred_points_pos'] = debug_model_position.predict(apply_set_features)
        apply_set['pred_points_all'] = debug_model_all.predict(apply_set_features)
        
        apply_set.to_csv('debug_' + position + '.csv')