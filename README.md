# Fantasy Football Draft Modeling Tool
Prepared for NFL fantasy leauges, this tool uses prior data to estimate expected points by player for the upcoming season. 

## Feautres:
- Scrape publicly available data for prior seasons (TODO: parameters to change historical data usage)
- Train a model based on prior performance for free agents (all players if pre-draft), with flexibility on which seasons and positions to use as a training set
- Save the model (TF format) for use without needing to re-query
- Load previously saved models and rank free agents based on projected points
- For ranking all players, output can be dumped to a csv
