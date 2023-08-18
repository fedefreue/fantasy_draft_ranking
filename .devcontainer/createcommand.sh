#!/bin/bash

echo "Installing required packages..."
sudo apt-get update
# sudo apt-get install 

echo "Installing Python dependencies..."
pip install -r requirements.txt