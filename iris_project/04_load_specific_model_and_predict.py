from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd


# load the model
#predictor = TabularPredictor.load('AutogluonModels/ag-20240612_132644/RandomForestEntr')
predictor = TabularPredictor.load('AutogluonModels/ag-20240612_132644')
model_names = predictor.model_names()
print("Model names: ", model_names)
print()

MODEL_TO_USE = "RandomForestGini"

# print a summary of how well it works
# results = predictor.fit_summary()

test_data = TabularDataset('../data/iris_test.csv')
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)

"""
Lines 19-29 below test our model on a single data point.
This data point is from the file data/iris_test.csv line #26

Here's the row, note that the correct prediction for species is 'Iris-virginica':
7.9,3.8,6.4,2.0,Iris-virginica

"""

# Define the column names as in the IRIS dataset
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Define the row as a list and omit the species column that we are trying to predict
row = [7.9, 3.8, 6.4, 2.0]

# Create a DataFrame
row_df = pd.DataFrame([row], columns=columns)

# predict the species of a single data point
correct_answer = 'Iris-virginica'
model_prediction= predictor.predict(row_df, model=MODEL_TO_USE)
print(f"The model predicted: {model_prediction[0]}")
print(f"The correct answer is: {correct_answer}")
