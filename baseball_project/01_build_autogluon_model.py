import pandas as pd
from autogluon.tabular import TabularPredictor

# Load the data
file_path = '../data/raw_data.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Columns to drop
drop_list = ['Y2=Pop 2','Y3=Pop 3','Y4=Pop 4','Y5=Pop 5','Y6= Pop 6']
data = data.drop(columns=drop_list)

# Remove columns following 'X96'
if 'X96' in data.columns:
    last_index = data.columns.get_loc('X96') + 1
    data = data.iloc[:, :last_index]

# Define the target variable and the features
target = 'Y1=Pop 1'
features = [col for col in data.columns if col != target and col != 'Observation_ID']

# Split the data into train and test sets
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

# Initialize the TabularPredictor
predictor = TabularPredictor(label=target)

# Train the model
predictor.fit(train_data[features + [target]])

# Evaluate the model
performance = predictor.evaluate(test_data[features + [target]])
print("Model Performance: ", performance)

# create a predictor object and fit it to the training data
# see also: https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html
# see also: https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html
predictor = TabularPredictor(label='species').fit(train_data=train_data, presets="medium_quality")

print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon identified the following types of features:")
print(predictor.feature_metadata)
print()

# make predictions on the test set
predictions = predictor.predict(test_data)
print("Predictions:")
print("************************************")
print("This prints the first 5 rows of the predictions:")
print(predictions.head(5))