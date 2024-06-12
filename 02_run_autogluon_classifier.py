import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

# Load the dataset
file_path = 'IRIS.csv'
iris_data = pd.read_csv(file_path)

# Split the data into training and testing sets
train_data, test_data = train_test_split(iris_data, test_size=0.2, random_state=42)

# Define the predictor
predictor = TabularPredictor(label='species', path='iris_model').fit(train_data)

# Make predictions on the test set
predictions = predictor.predict(test_data)

# Evaluate the model
performance = predictor.evaluate(test_data)
print(performance)
