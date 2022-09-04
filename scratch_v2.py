import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
import glob

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

layer1 = 13
layer2 = 6
thisYear = 2022
yearsToTrain = 10
tableForRank = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/2021.csv')
tableForRank = tableForRank.loc[tableForRank['Pos'].notnull()]

# Create an empty pd.DataFrame() called finalRanks
#finalRanks = pd.DataFrame()

def buildDataset(seasonYear: int, historyYears: int):
    df = pd.DataFrame()

    for i in range(historyYears):
        seasonYear -= 1
        df_new = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/' + str(seasonYear) + '.csv')
        df_new['seasonYear'] = seasonYear
        df = df.append(df_new)
    
    return df

def calculatePoints(rawStats: pd.DataFrame(), points: dict):
    rawStats['waffleHousePoints'] = points['Pass Yds'] * rawStats['PassingYds'] + points['Pass TD'] * rawStats['PassingTD'] + points['Int'] * rawStats['Int'] + points['Rush Att'] * rawStats['RushingAtt'] + points['Rush TD'] * rawStats['RushingTD'] + points['Rec'] * rawStats['Rec'] + points['Rec Yds'] * rawStats['ReceivingYds'] + points['Rec TD'] * rawStats['ReceivingTD'] + points['Fum Lost'] * rawStats['FumblesLost']

    # Remove rows from df where waffleHousePoints is 0
    rawStats = rawStats.loc[rawStats['waffleHousePoints'] != 0]

    # Remove * and + from Player
    rawStats['Player'] = rawStats['Player'].str.replace('*', '')
    rawStats['Player'] = rawStats['Player'].str.replace('+', '')

    # Add a column to rawStats that has that Player's waffleHousePoints for the row with seasonYear before it
    rawStats['waffleHousePointsBefore'] = rawStats.groupby('Player')['waffleHousePoints'].shift(1)

    # Remove rows where waffleHousePointsBefore is NaN
    rawStats = rawStats.loc[rawStats['waffleHousePointsBefore'].notnull()]

    return rawStats

def trainModel(layer1: int, layer2: int, modelFeatures, modelY):
    # Build a neural network using tensorflow to predict modelY using modelFeatures
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(layer1, activation=tf.nn.relu),
        tf.keras.layers.Dense(layer2, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
    ])

    # Run the model to predict modelY using modelFeatures and save the predictions to modelYPred
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(modelFeatures, modelY, epochs=20, verbose=0)
    modelYPred = model.predict(modelFeatures, verbose=0)

    # print(model.evaluate(modelFeatures, modelY))
    # print(model.summary())

    rSquare = r2_score(modelY, modelYPred, multioutput='variance_weighted')
    return rSquare, model

def createRanks(thisYear: int, yearsToTrain: int, points: dict, position: str, tableForRank: pd.DataFrame()):
    rawData = buildDataset(thisYear, yearsToTrain)
    forPrint = calculatePoints(rawData, points)

    tableForRank['Player'] = tableForRank['Player'].str.replace('*', '')
    tableForRank['Player'] = tableForRank['Player'].str.replace('+', '')
    tableForRank = tableForRank.loc[tableForRank['Pos'] == position]
    tableForRank_withPlayers = pd.DataFrame(tableForRank, columns = ['Player', 'Pos', 'Tgt', 'PassingAtt', 'RushingAtt', 'PassingYds', 'PassingTD', 'Int', 'RushingAtt', 'RushingTD', 'ReceivingYds', 'ReceivingTD', 'FumblesLost', 'Fumbles'])

    # Filter forPrint to only rows with Pos = position
    forPrint = forPrint.loc[forPrint['Pos'] == position]
    
    # Create a new pd.DataFrame with the columns we want
    modelFeatures = pd.DataFrame(forPrint, columns = ['Tgt', 'PassingAtt', 'RushingAtt', 'PassingYds', 'PassingTD', 'Int', 'RushingAtt', 'RushingTD', 'ReceivingYds', 'ReceivingTD', 'FumblesLost', 'Fumbles'])

    # modelFeatures = modelFeatures.loc[modelFeatures['Pos'] == position]
    modelY = forPrint['waffleHousePointsBefore']

    # Convert to numpy arrays
    modelFeatures = modelFeatures.to_numpy()
    modelY = modelY.to_numpy()

    # Remove NaN values from modelFeatures and modelY
    modelFeatures = modelFeatures[~np.isnan(modelFeatures).any(axis=1)]
    modelY = modelY[~np.isnan(modelY)]

    # Make a pd.DataFrame with three columns called evalTable
    evalTable = pd.DataFrame(columns=['layer1', 'layer2', 'rSquare'])
    best_r2 = 0.0

    for i in range(1, layer1 + 1):
        for j in range(1, min(i, layer2 + 1)):
            # Add a new row to evalTable, and add i + 1 in layer1 column and j +1 in layer2 column
            evalTable = evalTable.append(pd.Series([i, j, 0]), ignore_index=True)

            # In the last row of evalTable, add the output of trainModel(i + 1, j + 1, modelFeatures, modelY) to the third column
            evalTable.iloc[-1, 2], currentModel = trainModel(i, j, modelFeatures, modelY)

            # If the rSquare of the last row of evalTable is greater than best_r2, set best_r2 to that rSquare
            if evalTable.iloc[-1, 2] > best_r2:
                best_r2 = evalTable.iloc[-1, 2]
                bestModel = currentModel
                print('New Best: Layer 1 = ' + str(i) + ', Layer 2 = ' + str(j) + ', R2 = ' + str(evalTable.iloc[-1, 2]))
                
                # Remove columns Player and Pos from tableForRank
                tableForRank = tableForRank_withPlayers.drop(columns=['Player', 'Pos'])

                # Generate predictions with bestModel, andd to tableForRank
                tableForRank['predictedPoints'] = bestModel.predict(tableForRank, verbose=0)

                # Join tableForRank with tableForRank_withPlayers on index and add Player and Pos columns to tableForRank
                tableForRank = tableForRank.join(tableForRank_withPlayers[['Player', 'Pos']], how='left')

            j += 1
        i += 1

        # Rank tableForRank by predictedPoints descending
        # tableForRank = tableForRank.sort_values(by=['predictedPoints'], ascending=False)

        # Save tableForRank to a csv file called finalRank + position + .csv
        tableForRank.to_csv('finalRank_' + position + '.csv', index=False)

# createRanks(thisYear, yearsToTrain, points, position, tableForRank)

# For each position in tableToRank, run createRanks
for position in tableForRank['Pos'].unique():
     print('Ranking ' + position + '...')
     createRanks(thisYear, yearsToTrain, points, position, tableForRank)

# Find all finals with the name final_rank_*.csv and append them into a single table
finalTable = pd.concat([pd.read_csv(f) for f in glob.glob('finalRank_*.csv')], ignore_index = True) 

# Rank finalTable by predictedPoints descending
finalTable = finalTable.sort_values(by=['predictedPoints'], ascending=False)

# Save finalTable to a csv file called finalRank.csv
finalTable.to_csv('finalRank.csv', index=False)
