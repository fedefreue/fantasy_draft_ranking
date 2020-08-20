# C+ Team Draft Modeling Tool
Prepared for Yahoo Fantasy NFL leauges, this tool uses prior data to estimate expected points by player for the upcoming season. 

## Feautres:
- Connect to any NFL Yahoo Fantasy leauge, providing the league ID
- Train a model based on prior performance for free agents (all players if pre-draft), with flexibility on which seasons and positions to use as a training set
- Save the model to a Pickle container for use without needing to re-query Yahoo
- Load previously saved models from a Pickle container
- Rank free agents based on projected points

## Requirements
- Yahoo Fantasy account and enrollment in a league
- During the first connection, providing access to the package and providing the verification code in the terminal (no logging in after that is needed)