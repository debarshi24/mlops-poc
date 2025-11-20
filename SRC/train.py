import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression

# Load preprocessed training data
train = pd.read_csv("data/train.csv")

# Separate features and label
X_train = train.drop('churned', axis=1)
y_train = train['churned']

# Train a simple logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the trained model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.joblib")

print("✅ Model training complete. Model saved at model/model.joblib")