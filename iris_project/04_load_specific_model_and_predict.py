from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import os

#constants
MODEL_TO_USE = "RandomForestGini"


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
    # load the model
    # previously the model was hard-coded
    # predictor = TabularPredictor.load('AutogluonModels/ag-20240612_132644/RandomForestEntr')
    # now we will instead load the latest model
    model_path = get_latest_subdirectory('AutogluonModels')
    # load the model
    predictor = TabularPredictor.load("AutogluonModels/"+model_path)
    model_names = predictor.model_names()
    print("Models available: ", model_names)
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
    model_prediction= predictor.predict(row_df, model=MODEL_TO_USE)
    print(f"The model predicted: {model_prediction[0]}")
    print(f"The correct answer is: {correct_answer}")

if __name__ == '__main__':
    main()