#!/bin/bash

echo "Installing SQLite..."
sudo apt-get update
sudo apt-get install sqlite3

echo "Installing Python dependencies..."
pip install -r requirements.txt