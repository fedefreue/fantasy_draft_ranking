
import pandas as pd
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt


# Get data for the last 5 years
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/2019.csv')



points = {
    'Pass Yds':     0.04,
    'Pass TD':      4,
    'Int':          -1,
    'Rush Att':     0.1,
    'Rush TD':      6,
    'Ret TD':       6,
    'Rec':          0.5,
    'Rec Yds':      0.1,
    'Rec TD':       6,
    '2-PT':         2,
    'Fum Lost':     -2,
    'Fum Ret TD':   2
}

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


rawData = buildDataset(2022, 5)
forPrint = calculatePoints(rawData, points)
print(forPrint)

# Show all rows in forPrint where Player is Cooper Kupp
print(forPrint.loc[forPrint['Player'] == 'Cooper Kupp'])

# Create a new pd.DataFrame with the columns we want
modelFeatures = pd.DataFrame(forPrint, columns = ['PassingYds', 'PassingTD', 'Int', 'RushingAtt', 'RushingTD', 'ReceivingYds', 'ReceivingTD', 'FumblesLost'])
modelY = forPrint['waffleHousePointsBefore']

print(modelFeatures)
print(modelY)

modelFeatures = modelFeatures.to_numpy()
modelY = modelY.to_numpy()


# Remove NaN values from modelFeatures and modelY
modelFeatures = modelFeatures[~np.isnan(modelFeatures).any(axis=1)]
modelY = modelY[~np.isnan(modelY)]

model = sklearn.linear_model.LinearRegression().fit(modelFeatures, modelY)
print(model.score(modelFeatures, modelY))
plt.scatter(model.predict(modelFeatures), modelY)
plt.show()

# print(forPrint)

#   features = masterTable.drop(columns = ['position_type', 'player_id', 'Points'])
#         y_train = masterTable['Points']

#         features = features.to_numpy()
#         y_train = y_train.to_numpy()

#         self.model = sklearn.linear_model.LinearRegression().fit(features, y_train)
#         #print(self.model.score(features, y_train))
#         self.statusText.set('R2: ' + str(self.model.score(features, y_train))) 
#         plt.scatter(self.model.predict(features), y_train)
#         plt.show()