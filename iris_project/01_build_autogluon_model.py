from autogluon.tabular import TabularDataset, TabularPredictor


# load two data sets:
#  - "train_data" for training 
#  -  "test_data" for testing
train_data = TabularDataset('../data/iris_train.csv')
test_data = TabularDataset('../data/iris_test.csv')

# create a predictor object and fit it to the training data
predictor = TabularPredictor(label='species').fit(train_data=train_data)

# make predictions on the test set
#predictions = predictor.predict(test_data)
#print(predictions)

# display the model's performance
results = predictor.fit_summary()

# show the model leaderboard
leaderboard = predictor.leaderboard(test_data)
#print(leaderboard)