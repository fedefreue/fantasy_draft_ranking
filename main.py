from tkinter import Tk, ttk
import tkinter as tk
import connect
import playerData
import yahoo_fantasy_api as yfa
import sklearn.linear_model
import pandas as pd
import matplotlib.pyplot as plt
import rankPlayers

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

class GUI(object):
    def __init__(self, master):
        self.master = master
        master.title("Fed Draft Analysis Tool")
        master.minsize(400,400)

        self.connectElementsColumn(columnVar = 0)
        self.trainElementsColumn(columnVar = 1)
        self.applyElementsColumn(columnVar = 2)

        self.statusText = tk.StringVar()
        self.statusTextLabel = tk.Label(self.master, textvariable = self.statusText)
        self.statusTextLabel.grid(column = 0, row = 5)

    def connectElementsColumn(self, columnVar: int):
        self.connectFile = tk.StringVar()
        self.connectFile.set('private.json')
        
        self.connectButton = ttk.Button(self.master, text = 'Connect', command = self._connect)
        self.connectFileEntry = ttk.Entry(self.master, width = 20, textvariable = self.connectFile)
        
        self.connectFileEntry.grid(column = columnVar, row = 1)
        self.connectButton.grid(column = columnVar, row = 3)

    def applyElementsColumn(self, columnVar: int):
        self.applyPositionList = tk.StringVar()
        self.applyPositionList.set('Positions')

        self.applyInputLabel = ttk.Label(self.master, text = 'Apply Model and Rank')
        self.applyButton = ttk.Button(self.master, text = 'Rank Players', command = self._rankPlayers)
        self.applyPositionEntry = ttk.Entry(self.master, width = 20, textvariable = self.applyPositionList)

        self.applyInputLabel.grid(column = columnVar, row = 0)
        self.applyPositionEntry.grid(column = columnVar, row = 1)
        self.applyButton.grid(column = columnVar, row = 3)

    def trainElementsColumn(self, columnVar: int):
        self.trainSeasonList = tk.StringVar()
        self.trainSeasonList.set('Seasons')
        self.trainPositionList = tk.StringVar()
        self.trainPositionList.set('Positions')

        self.trainButton = ttk.Button(self.master, text = 'Train Model', command = self._trainModel)
        self.trainInputLabel = ttk.Label(self.master, text = 'Training Inputs')
        self.trainEntrySeasonList = ttk.Entry(self.master, width = 20, textvariable = self.trainSeasonList)
        self.trainPositionsList = ttk.Entry(self.master, width = 20, textvariable = self.trainPositionList)

        self.trainInputLabel.grid(column = columnVar, row = 0)
        self.trainEntrySeasonList.grid(column = columnVar, row = 1)
        self.trainPositionsList.grid(column = columnVar, row = 2)
        self.trainButton.grid(column = columnVar, row = 3)

        #result=textExample.get("1.0", "end")

    def _connect(self):
        self.statusText.set('Connecting...')
        fileName = self.connectFileEntry.get()
        self.connection = connect.connect(fileName)
        self.league = yfa.League(self.connection,'nfl.l.254924')
        self.statusText.set('Connected to Yahoo Fantasy!')

    def _trainModel(self):
        seasonList = self.trainEntrySeasonList.get()
        seasonList = seasonList.split(',')
        
        positionList = self.trainPositionsList.get()
        positionList = positionList.split(',')
        
        self.statusText.set('Training model for seasons ' + str(seasonList) + ' and positions ' + str(positionList) + '...')

        masterTable = playerData.generateTrainingTable(seasonList, self.league, points, positionList)

        self.statusText.set('Master table created. Training model...')

        features = masterTable.drop(columns = ['position_type', 'player_id', 'Points'])
        y_train = masterTable['Points']

        features = features.to_numpy()
        y_train = y_train.to_numpy()

        self.model = sklearn.linear_model.LinearRegression().fit(features, y_train)
        #print(self.model.score(features, y_train))
        self.statusText.set('R2: ' + str(self.model.score(features, y_train))) 
        plt.scatter(self.model.predict(features), y_train)
        plt.show()

    def _rankPlayers(self):
        position = self.applyPositionEntry.get()
        self.statusText.set('Applying model for position ' + str(position) + '...')
        
        staging = rankPlayers.rankPlayers(position, self.league, self.model, '2019')
        self.statusText.set('Writing to file...')
        staging.to_csv('test_file.csv')

root = Tk()
example = GUI(root)
root.mainloop()

