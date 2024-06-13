import os
from autogluon.tabular import TabularDataset, TabularPredictor
import warnings
warnings.filterwarnings("ignore", message="AutoGluon summary plots cannot be created because bokeh is not installed.")

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
    print("Loading model...")
    # get the latest model directory
    BASE_DIR = 'AutogluonModels'
    model_dir = get_latest_subdirectory(BASE_DIR)
    most_recent_model_dir = BASE_DIR+"/%s" %(model_dir)
    print("Most recent model directory: %s" % most_recent_model_dir )
    predictor = TabularPredictor.load(most_recent_model_dir)
    print("Done loading model from (%s)." % most_recent_model_dir)

    # display the model's performance
    print("Model Performance:")
    print("************************************")
    print()
    _ = predictor.fit_summary() # the underscore variable just means "ignore the output of this function call"
    #print(results)
    

    # show the model leaderboard
    print("Model Leaderboard:")
    print("************************************")
    print()
    test_data = TabularDataset('../data/raw_data_test.csv')
    leaderboard = predictor.leaderboard(test_data)
    print(leaderboard)
    print()


if __name__ == '__main__':
    main()