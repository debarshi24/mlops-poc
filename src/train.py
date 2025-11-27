# src/train.py
import os
import io
import joblib
import boto3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def _read_csv(maybe_s3_uri):
    """
    Read CSV from local path or s3://bucket/key using boto3.
    Returns a pandas.DataFrame.
    """
    if maybe_s3_uri is None:
        raise ValueError("data_path is None. Provide a local path or s3:// URI.")
    if maybe_s3_uri.startswith("s3://"):
        # parse s3://bucket/key
        uri = maybe_s3_uri[5:]
        bucket, key = uri.split("/", 1)
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj['Body'].read()))
    else:
        # local file
        return pd.read_csv(maybe_s3_uri)

def train_model(data_path=None, model_output_path="model.joblib", target_col="target", test_size=0.2, random_state=42, **kwargs):
    """
    Train a simple LogisticRegression pipeline and save it.

    Args:
        data_path (str): Local CSV path or s3:// URI to CSV file. If None, tries env var DATA_S3_URI.
        model_output_path (str): Where to save the trained model (local path).
        target_col (str): Name of the target column in the CSV.
        test_size (float): Fraction for test split.
        random_state (int): Random seed.
        **kwargs: any additional args (ignored).
    Returns:
        dict with keys: {'model_path', 'train_accuracy'}
    """
    # fallback to env var if not provided
    if data_path is None:
        data_path = os.environ.get("DATA_S3_URI") or os.environ.get("DATA_PATH")
    if not data_path:
        raise ValueError("No data_path provided and DATA_S3_URI / DATA_PATH env var not set.")

    df = _read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. Columns: {list(df.columns)}")

    # Basic preprocessing: drop rows with NA in features/target
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # If any non-numeric columns exist, try simple conversion (user should replace with better preprocessing)
    # Convert object columns to numeric where possible (coerce errors -> NaN, then drop)
    for c in X.select_dtypes(include=["object", "category"]).columns:
        try:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        except Exception:
            pass
    X = X.dropna(axis=1, how="all")  # drop columns that became all-NaN
    X = X.dropna()  # drop rows with NaNs produced by coercion

    # align y with the filtered X
    y = y.loc[X.index]

    # Quick train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y)) > 1 else None
    )

    # Pipeline: scaler + logistic regression (tweak params as needed)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)

    # Evaluate on test set
    preds = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

    # Ensure output directory exists
    out_dir = os.path.dirname(model_output_path) or "."
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    joblib.dump(pipe, model_output_path)

    return {"model_path": model_output_path, "train_accuracy": acc}

# --- compatibility shim: expose train_model for callers that expect that name ---
# If your file already defines a function called `train` (or `run` / `main`), this
# wrapper will call it. If not, it will do nothing and leave the file unchanged.
if 'train_model' not in globals():
    if 'train' in globals():
        def train_model(*args, **kwargs):
            return train(*args, **kwargs)
    elif 'run' in globals():
        def train_model(*args, **kwargs):
            return run(*args, **kwargs)
    elif 'main' in globals():
        def train_model(*args, **kwargs):
            return main(*args, **kwargs)
# --- end shim ---
