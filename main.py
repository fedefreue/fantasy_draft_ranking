
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Tk, ttk

import pandas as pd
import sklearn.linear_model
import yahoo_fantasy_api as yfa
import pickle

import connect
import playerData
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
        master.geometry('500x300')
        master.minsize(500,300)

        self.connectFrame = ttk.Frame(self.master)
        self.connectFrame.grid(row = 0, column = 0)

        self.trainFrame = ttk.Frame(self.master)
        self.trainFrame.grid(row = 1)

        self.applyFrame = ttk.Frame(self.master)
        self.applyFrame.grid(row = 2)

        self.statusFrame = ttk.Frame(self.master)
        self.statusFrame.grid(row = 3)

        self.connectElementsFrame(frame = self.connectFrame)
        self.trainElementsFrame(frame = self.trainFrame)
        self.applyElementsFrame(frame = self.applyFrame)

        self.statusText = tk.StringVar()
        self.statusTextLabel = tk.Label(self.statusFrame, textvariable = self.statusText)
        self.statusTextLabel.pack()

    def connectElementsFrame(self, frame: ttk.Frame):
        self.connectTitle = ttk.Label(frame, text = 'Connect to Yahoo', borderwidth = 5)
        self.tokenLabel = ttk.Label(frame, text = 'Token File:', anchor = 'e',width=12)
        self.connectFile = tk.StringVar()
        self.connectFile.set('private.json')
        
        self.connectButton = ttk.Button(frame, text = 'Connect', command = self._connect, width = 8)
        self.connectFileEntry = ttk.Entry(frame, width = 20, textvariable = self.connectFile)
        
        self.leagueLabel = ttk.Label(frame, text = 'Leauge ID:', anchor = 'e', width=12)
        self.leaugeID = tk.StringVar()
        self.leaugeID.set('254924')
        self.leagueEntry = ttk.Entry(frame, width = 20, textvariable = self.leaugeID)

        self.tokenLabel.grid(column = 0, row = 1)
        self.connectTitle.grid(column = 1, row = 0)
        self.connectFileEntry.grid(column = 1, row = 1)
        self.connectButton.grid(column = 2, row = 1)
        self.leagueLabel.grid(column = 0, row = 2)
        self.leagueEntry.grid(column = 1, row = 2)

    def applyElementsFrame(self, frame: ttk.Frame):
        self.applyPositionList = tk.StringVar()
        self.applyPositionList.set('Positions')
        self.fileNameEntry = tk.StringVar()
        self.fileNameEntry.set('model.pkl')

        self.applyInputLabel = ttk.Label(frame, text = 'Model Management', borderwidth = 5)
        self.applyButton = ttk.Button(frame, text = 'Rank Players', command = self._rankPlayers, width = 8)
        self.applyPositionEntry = ttk.Entry(frame, width = 20, textvariable = self.applyPositionList)
        self.applyLabel = ttk.Label(frame, text = 'Positions:', anchor = 'e',width=12)

        self.loadButton = ttk.Button(frame, text = 'Load Model', command = self._loadModel, width = 8)
        self.saveButton = ttk.Button(frame, text = 'Save Model', command = self._saveModel, width = 8)
        self.saveLabel = ttk.Label(frame, text = 'Model File Name:', anchor = 'e',width=12)
        self.saveEntry = ttk.Entry(frame, width = 20, textvariable = self.fileNameEntry)


        self.applyInputLabel.grid(column = 1, row = 0)
        self.applyPositionEntry.grid(column = 1, row = 1)
        self.applyButton.grid(column = 2, row = 3)
        self.saveLabel.grid(column = 0, row = 2)
        self.saveEntry.grid(column = 1, row = 2)
        self.loadButton.grid(column = 2, row = 1)
        self.saveButton.grid(column = 2, row = 2)
        self.applyLabel.grid(column = 0, row = 1)

    def trainElementsFrame(self, frame: ttk.Frame):
        self.trainSeasonList = tk.StringVar()
        self.trainSeasonList.set('Seasons')
        self.trainPositionList = tk.StringVar()
        self.trainPositionList.set('Positions')

        self.seasonsLabel = ttk.Label(frame, text = 'Seasons:', anchor = 'e',width=12)
        self.PositionsLabel = ttk.Label(frame, text = 'Positions:', anchor = 'e',width=12)
        self.trainButton = ttk.Button(frame, text = 'Train Model', command = self._trainModel, width = 8)
        self.trainInputLabel = ttk.Label(frame, text = 'Training Inputs', borderwidth = 5)
        self.trainEntrySeasonList = ttk.Entry(frame, width = 20, textvariable = self.trainSeasonList)
        self.trainPositionsList = ttk.Entry(frame, width = 20, textvariable = self.trainPositionList)

        self.seasonsLabel.grid(column = 0, row = 1)
        self.PositionsLabel.grid(column = 0, row = 2)
        self.trainInputLabel.grid(column = 1, row = 0)
        self.trainEntrySeasonList.grid(column = 1, row = 1)
        self.trainPositionsList.grid(column = 1, row = 2)
        self.trainButton.grid(column = 2, row = 1)

    def _connect(self):
        self.statusText.set('Connecting...')
        fileName = self.connectFileEntry.get()
        self.connection = connect.connect(fileName)
        self.leagueCode = str('nfl.l.' + self.leaugeID.get())
        self.league = yfa.League(self.connection, self.leagueCode) #'nfl.l.254924'
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

    def _saveModel(self):
        self.pkl_filename = self.fileNameEntry.get()
        with open(self.pkl_filename, 'wb') as file:
            pickle.dump(self.model, file)
        
        self.statusText.set('Saved model to ' + str(self.pkl_filename))

    def _loadModel(self):
        self.pkl_filename = self.fileNameEntry.get()
        with open(self.pkl_filename, 'rb') as file:
            self.model = pickle.load(file)
        
        self.statusText.set('Loaded ' + str(self.pkl_filename))
        
    def _rankPlayers(self):
        position = self.applyPositionEntry.get()
        self.statusText.set('Applying model for position ' + str(position) + '...')
        
        staging = rankPlayers.rankPlayers(position, self.league, self.model, '2019')
        self.statusText.set('Writing to file...')
        staging.to_csv('test_file.csv')
        self.statusText.set('Output file complete!')

root = Tk()
example = GUI(root)
root.mainloop()