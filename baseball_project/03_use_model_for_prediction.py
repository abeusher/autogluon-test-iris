from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import os

def list_subdirectories(parent_dir, prefix='ag-2024'):
    """get a list of subdirectories in the parent directory that start with the prefix 'ag-2024' aka autogluon-2024"""
    subdirectories = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith(prefix)]
    return subdirectories

def get_latest_subdirectory(parent_dir, prefix='ag-2024'):
    """get the latest subdirectory in the parent directory that start with the prefix 'ag-2024' aka autogluon-2024"""
    subdirectories = list_subdirectories(parent_dir, prefix)
    print("subdirectories: ", subdirectories)
    latest_subdirectory = max(subdirectories)
    print("latest_subdirectory: ", latest_subdirectory)
    return latest_subdirectory

def main():
    model_path = get_latest_subdirectory('AutogluonModels')
    # load the model
    predictor = TabularPredictor.load("AutogluonModels/"+model_path)

    # print a summary of how well it works
    # results = predictor.fit_summary()

    test_data = TabularDataset('../data/raw_data_test.csv')
    leaderboard = predictor.leaderboard(test_data)
    print(leaderboard)
    print()

    # Outout the feature importance
    feature_importance = predictor.feature_importance(test_data)
    print(feature_importance)
    print()

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
    model_prediction= predictor.predict(row_df)
    print("Evaluating the model on a single data point from data/iris_test.csv")
    print("input values: %s" %(row_df.values[0]))
    print(f"The model predicted: {model_prediction[0]}")
    print(f"The correct answer is: {correct_answer}")


if __name__ == '__main__':
    main()