from autogluon.tabular import TabularDataset, TabularPredictor

# load the model
predictor = TabularPredictor.load('AutogluonModels/ag-20240612_155114')

# display the model's performance
results = predictor.fit_summary()

# show the model leaderboard
test_data = TabularDataset('../data/iris_test.csv')
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)