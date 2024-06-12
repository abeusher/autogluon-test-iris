from autogluon.tabular import TabularDataset, TabularPredictor
import warnings
warnings.filterwarnings("ignore", message="AutoGluon summary plots cannot be created because bokeh is not installed.")

# load the model
print("Loading model...")
predictor = TabularPredictor.load('AutogluonModels/ag-20240612_155114')
print("Done loading model.")

# display the model's performance
results = predictor.fit_summary()

# show the model leaderboard
print("Model Leaderboard:")
print("************************************")
print()
test_data = TabularDataset('../data/iris_test.csv')
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)
print()