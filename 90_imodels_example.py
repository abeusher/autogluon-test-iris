import pandas as pd
from sklearn.model_selection import train_test_split
from imodels import HSTreeClassifierCV
from sklearn.metrics import classification_report, accuracy_score, log_loss

# Load the training and testing data
train_data = pd.read_csv('data/iris_train.csv')
test_data = pd.read_csv('data/iris_test.csv')

# Prepare the data
X_train = train_data.drop(columns=['species']).values
y_train = train_data['species'].values
X_test = test_data.drop(columns=['species']).values
y_test = test_data['species'].values

# Fit the model
model = HSTreeClassifierCV(max_leaf_nodes=4)  # Initialize a tree model and specify only 4 leaf nodes
model.fit(X_train, y_train)   # Fit model

# Make predictions
preds = model.predict(X_test) # Discrete predictions: shape is (n_test, 1)
preds_proba = model.predict_proba(X_test) # Predicted probabilities: shape is (n_test, n_classes)

# Print the model
print(model)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, preds))
print("Accuracy:", accuracy_score(y_test, preds))
print("Log Loss:", log_loss(y_test, preds_proba))

# Display predictions and predicted probabilities
print("Predictions:", preds)
print("Predicted Probabilities:", preds_proba)
