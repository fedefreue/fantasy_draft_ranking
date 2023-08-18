import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn.metrics import r2_score
import tensorflow as tf
import tensorflow_ranking as tfr
import data_prep
import sqlite3

layer1 = 13
layer2 = 6
current_year = 2022
years_to_train = 15

# Read the table features from db.sqlite3 and load it into a pandas dataframe
# df = pd.read_sql("SELECT * FROM features", dbConnection)


def data_gen_year_list(start_year, num_years):
    year_list = []
    for i in range(num_years):
        year_list.append(start_year - i)
    return year_list


def model_train(layer1: int, layer2: int, modelFeatures, modelY):
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


def model_ranks(thisYear: int, yearsToTrain: int):
    data_years = data_gen_year_list(thisYear, yearsToTrain)

    data_prep.data_rawdl(data_years, dbConnection)
    stat_staging = data_prep.data_format(dbConnection)
    stat_staging.to_csv("debug_staging.csv", index=False)

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

    # Convert to numpy arrays
    stat_features = stat_features.to_numpy()
    stat_y = stat_y.to_numpy()

    stat_features = stat_features[~np.isnan(stat_features).any(axis=1)]
    stat_y = stat_y[~np.isnan(stat_y)]

    # Make a pd.DataFrame with three columns called evalTable
    evalTable = pd.DataFrame(columns=["layer1", "layer2", "rSquare"])
    best_r2 = 0.0

    lr_model = sklearn.linear_model.LinearRegression().fit(stat_features, stat_y)
    best_r2 = lr_model.score(stat_features, stat_y)
    print("R2: " + str(best_r2))

    stat_staging_best = stat_staging
    stat_staging_best["pred_points"] = lr_model.predict(stat_features)

    for i in range(1, layer1 + 1):
        for j in range(1, min(i, layer2 + 1)):
            print("Layer 1 = " + str(i) + ", Layer 2 = " + str(j))

            stat_features_iter = stat_features
            stat_staging_iter = stat_staging

            evalTable = pd.concat([evalTable, pd.Series([i, j, 0])], ignore_index=True)  # type: ignore
            # evalTable = evalTable.append(pd.Series([i, j, 0]), ignore_index=True)

            # In the last row of evalTable, add the output of trainModel(i + 1, j + 1, modelFeatures, modelY) to the third column
            evalTable.iloc[-1, 2], currentModel = model_train(
                i, j, stat_features, stat_y
            )

            # If the rSquare of the last row of evalTable is greater than best_r2, set best_r2 to that rSquare
            if evalTable.iloc[-1, 2] > best_r2:
                best_r2 = evalTable.iloc[-1, 2]
                bestModel = currentModel
                print(
                    "New Best: Layer 1 = "
                    + str(i)
                    + ", Layer 2 = "
                    + str(j)
                    + ", R2 = "
                    + str(evalTable.iloc[-1, 2])
                )

                stat_staging_best = stat_staging
                # Generate predictions with bestModel
                stat_staging_best["pred_points"] = bestModel.predict(
                    stat_features_iter, verbose=1
                )

            j += 1
        i += 1

        stat_staging_best.to_csv("debug.csv", index=False)
        return stat_staging_best


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

model_ranks(current_year, years_to_train)