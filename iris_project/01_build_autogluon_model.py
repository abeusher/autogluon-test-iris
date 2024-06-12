from autogluon.tabular import TabularDataset, TabularPredictor

# load two data sets:
#  - "train_data" for training 
#  -  "test_data" for testing
train_data = TabularDataset('../data/iris_train.csv')
test_data = TabularDataset('../data/iris_test.csv')

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