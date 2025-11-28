# src/model/inference.py

import joblib
import os
import json
import pandas as pd
from io import StringIO

"""
SageMaker SKLearn container expects the following functions:

- model_fn(model_dir)
- input_fn(request_body, content_type)
- predict_fn(input_data, model)
- output_fn(prediction, accept)

This file defines how the deployed endpoint:
1. Loads your trained model (model.joblib inside model.tar.gz)
2. Parses incoming requests (CSV or JSON)
3. Makes predictions
4. Returns a JSON response
"""

def model_fn(model_dir):
    """
    Load the trained model from the SageMaker model directory.
    SageMaker automatically unpacks model.tar.gz into model_dir.
    """
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, content_type="text/csv"):
    """
    Convert the incoming request payload to a Pandas DataFrame.

    Supports:
      - text/csv
      - application/json
    """
    if content_type == "text/csv":
        # Expect raw CSV with or without header
        s = StringIO(request_body)
        df = pd.read_csv(s)
        return df

    elif content_type == "application/json":
        data = json.loads(request_body)

        # If the input is a dict → convert to single-row dataframe
        if isinstance(data, dict):
            return pd.DataFrame([data])

        # If it's a list of dicts or list of lists
        return pd.DataFrame(data)

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Perform prediction using the trained model.

    Performs one-hot encoding using get_dummies() to match training.
    (Note: Must align features to trained model if missing columns.)
    """
    # One-hot encode input
    if isinstance(input_data, pd.DataFrame):
        X = pd.get_dummies(input_data)
    else:
        X = input_data

    # Try direct prediction
    try:
        preds = model.predict(X)
        return preds
    except Exception:
        # If training included dummies with certain columns,
        # align missing columns dynamically.
        if hasattr(model, "feature_names_in_"):
            model_cols = list(model.feature_names_in_)
            for c in model_cols:
                if c not in X.columns:
                    X[c] = 0
            X = X[model_cols]

        preds = model.predict(X)
        return preds


def output_fn(prediction, accept="application/json"):
    """
    Format the prediction output.
    """
    return json.dumps({"predictions": prediction.tolist()}), accept