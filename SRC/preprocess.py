import pandas as pd
from sklearn.model_selection import train_test_split
import os




def preprocess_csv(input_csv_path, output_prefix='/tmp'):
"""Load CSV, do simple cleaning, and write train/test CSVs to output_prefix.
Returns (train_path, test_path)
"""
df = pd.read_csv(input_csv_path)


# Basic cleaning: drop duplicates and rows with all-NaN
df = df.drop_duplicates().dropna(how='all')


# For POC: assume target column named 'target' exists; if not, try to infer last column
if 'target' not in df.columns:
target_col = df.columns[-1]
else:
target_col = 'target'


# Simple fillna for numeric columns
for c in df.select_dtypes(include=['number']).columns:
df[c] = df[c].fillna(df[c].median())


# Categorical fill
for c in df.select_dtypes(include=['object', 'category']).columns:
df[c] = df[c].fillna('')


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col] if target_col in df else None)


os.makedirs(output_prefix, exist_ok=True)
train_path = os.path.join(output_prefix, 'train.csv')
test_path = os.path.join(output_prefix, 'test.csv')
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
return train_path, test_path