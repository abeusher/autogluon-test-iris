from sklearn.model_selection import train_test_split

"""
Simple script to split the IRIS dataset into training and testing sets.

I obtained the data file from: https://www.kaggle.com/datasets/arshid/iris-flower-dataset
"""

# Split the data into training and testing sets
train_data, test_data = train_test_split(iris_data, test_size=0.2, random_state=42)

# Save the split data into separate CSV files
train_data_path = '/mnt/data/IRIS_train.csv'
test_data_path = '/mnt/data/IRIS_test.csv'

train_data.to_csv(train_data_path, index=False)
test_data.to_csv(test_data_path, index=False)

train_data_path, test_data_path
