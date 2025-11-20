import pandas as pd
import os

# Input and output paths
input_path = "s3://mlops-poc-artifacts/data/Topic_15_poc_customers.csv"
os.makedirs("data", exist_ok=True)

# Read dataset
df = pd.read_csv(input_path)

# Basic cleaning
df = df.dropna()
df = pd.get_dummies(df, columns=['plan'], drop_first=True)

# Split into train/test
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Save processed files locally (CodeBuild artifact)
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("✅ Preprocessing complete. Train/Test files ready.")
