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

    data_file = '../data/raw_data_test.csv'
    test_data = TabularDataset(data_file)
    leaderboard = predictor.leaderboard(test_data)
    print(leaderboard)
    print()

    # Outout the feature importance
    feature_importance = predictor.feature_importance(test_data)
    print(feature_importance)
    print()

    # Define the column names as in the IRIS dataset
    columns = ['Observation_ID', 'Y1=Pop 1',  'X11', 'X12', 'X13', 'X14', 'X15', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26', 'X31', 'X32', 
           'X33', 'X34', 'X35', 'X36', 'X41', 'X42', 'X43', 'X44', 'X51', 'X52', 'X53', 'X54', 'X61', 
           'X62', 'X63', 'X64', 'X65', 'X66', 'X67', 'X71', 'X72', 'X73', 'X74', 'X81', 'X82', 'X83', 
           'X84', 'X85', 'X91', 'X92', 'X93', 'X94', 'X95', 'X96']

    # Define the row as a list and omit the species column that we are trying to predict
    row = [15396,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0]

    # Create a DataFrame
    row_df = pd.DataFrame([row], columns=columns)

    # predict the species of a single data point
    correct_answer = 0
    model_prediction= predictor.predict(row_df)
    print("Evaluating the model on a single data point from %s" % data_file)
    print("input values: %s" %(row_df.values[0]))
    print(f"The model predicted: {model_prediction[0]}")
    print(f"The correct answer is: {correct_answer}")


if __name__ == '__main__':
    main()