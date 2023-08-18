import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import tensorflow as tf
import sqlite3

layer1 = 13
layer2 = 6
current_year = 2022
years_to_train = 15

# Read the table features from db.sqlite3 and load it into a pandas dataframe
# df = pd.read_sql("SELECT * FROM features", dbConnection)


def model_train_tf(layer1: int, layer2: int, modelFeatures, modelY):
    # Build a neural network using tensorflow to predict modelY using modelFeatures
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(layer1, activation=tf.nn.relu),
            tf.keras.layers.Dense(layer2, activation=tf.nn.relu),
            tf.keras.layers.Dense(1),
        ]
    )

    # Run the model to predict modelY using modelFeatures and save the predictions to modelYPred
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(modelFeatures, modelY, epochs=20, verbose=0)
    modelYPred = model.predict(modelFeatures, verbose=0)

    # print(model.evaluate(modelFeatures, modelY))
    # print(model.summary())

    rSquare = r2_score(modelY, modelYPred, multioutput="variance_weighted")
    return rSquare, model

def model_train_lr(stat_features, stat_y):
    model = tf.keras.experimental.LinearModel()

    model.compile(optimizer="adam", loss="mse")    
    model.fit(stat_features, stat_y, epochs=20)
    stat_y_pred = model.predict(stat_features, verbose=0)
    
    best_r2 = r2_score(stat_y, stat_y_pred, multioutput="variance_weighted")
    
    # lr_model = sklearn.linear_model.LinearRegression().fit(stat_features, stat_y)
    # best_r2 = lr_model.score(stat_features, stat_y)
    
    print("Baseline R2: " + str(best_r2))
    return best_r2, model

def model_ranks(db_connection: sqlite3.Connection, staging_table_name: str, save_model: int):
    stat_staging = pd.read_sql_query("SELECT * FROM " + str(staging_table_name), db_connection)

    # Remove any row with a NaN value
    stat_staging = stat_staging.dropna(axis=0)

    stat_features = pd.DataFrame(
        stat_staging,
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
    stat_y = stat_staging["points"]

    # Convert to numpy arrays and remove NaN
    stat_features = stat_features.to_numpy()
    stat_y = stat_y.to_numpy()
    stat_features = stat_features[~np.isnan(stat_features).any(axis=1)]
    stat_y = stat_y[~np.isnan(stat_y)]

    evalTable = pd.DataFrame(columns=["layer1", "layer2", "rSquare"])
    best_r2 = 0.0
    best_arch = None
    best_model = None

    # Baseline LR as a benchmark
    best_r2, best_model = model_train_lr(stat_features, stat_y)

    stat_staging_best = stat_staging
    stat_staging_best["pred_points"] = best_model.predict(stat_features)

    for i in range(1, layer1 + 1):
        for j in range(1, min(i, layer2 + 1)):
            stat_features_iter = stat_features
            # stat_staging_iter = stat_staging

            evalTable = pd.concat([evalTable, pd.Series([i, j, 0])], ignore_index=True)  # type: ignore

            # In the last row of evalTable, add the output of trainModel(i + 1, j + 1, modelFeatures, modelY) to the third column
            evalTable.iloc[-1, 2], current_model = model_train_tf(
                i, j, stat_features, stat_y
            )

            # If the rSquare of the last row of evalTable is greater than best_r2, set best_r2 to that rSquare
            if evalTable.iloc[-1, 2] > best_r2:
                best_r2 = evalTable.iloc[-1, 2]
                best_model = current_model
                best_arch = "New Best: Layer 1 = " + str(i) + ", Layer 2 = " + str(j) + ", R2 = " + str(evalTable.iloc[-1, 2])
                
                stat_staging_best = stat_staging
                stat_staging_best["pred_points"] = best_model.predict(
                    stat_features_iter, verbose=1
                )

            j += 1
        i += 1

    if save_model == 1:
        tf.keras.saving.save_model(best_model, "model.tf", overwrite = True, save_format = "tf")
        # best_model.save_model('model.keras')

    return best_arch


""" 
dbConnection = sqlite3.connect("db.sqlite3")

# Check if the initializated bit exists
if (
    dbConnection.execute("SELECT COUNT(*) FROM dbInitialize WHERE bool = 1").fetchall()
) is not None:
    print("Database has been initialized")
else:
    raise Exception("Database has not been initialized")

# For each value in dataYears, scrape the data into a raw_data table
dbConnection.execute("DROP TABLE IF EXISTS rawData")
dbConnection.commit()

model_ranks(dbConnection, "features", 0)
 """