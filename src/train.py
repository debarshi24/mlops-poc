# src/train.py
import os
import logging
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

def train_model(data_path: str = None, output_dir: str = "models"):
    """
    Minimal train_model entrypoint expected by ml_pipeline.py.
    - data_path: local path or S3 path already downloaded by CI
    - output_dir: where to write model.joblib
    """
    # fallback sample: look for DATA_S3_URI or a local file
    if data_path is None:
        data_path = os.environ.get("DATA_LOCAL_PATH") or os.environ.get("DATA_S3_URI")
    if not data_path:
        raise ValueError("No data path provided. Set data_path arg or DATA_LOCAL_PATH/DATA_S3_URI env var.")

    logger.info("Loading data from %s", data_path)
    # if it's a CSV local path:
    df = pd.read_csv(data_path)

    # simple example: require the CSV to have 'label' and feature columns
    if "label" not in df.columns:
        raise ValueError("Training CSV must contain a 'label' column")

    X = df.drop(columns=["label"])
    y = df["label"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    logger.info("Fitting model on %d rows x %d cols", X.shape[0], X.shape[1])
    pipeline.fit(X, y)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(pipeline, model_path)
    logger.info("Saved model to %s", model_path)
    return model_path


# keep the "compatibility shim" so older callers still work
if __name__ == "__main__":
    # If run as a script, expect a local CSV path in env var or default file
    data_path = os.environ.get("DATA_LOCAL_PATH", "data/raw/Topic_15_poc_customers.csv")
    out = train_model(data_path=data_path)
    print("Trained and saved model:", out)
