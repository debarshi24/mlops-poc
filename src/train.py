# src/train.py
import os
import logging
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from urllib.parse import urlparse
import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

def parse_s3_uri(s3_uri):
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError("Not an s3 uri")
    return parsed.netloc, parsed.path.lstrip("/")

def download_s3_to_local(s3_uri, local_path):
    bucket, key = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    logger.info("Downloading %s to %s", s3_uri, local_path)
    s3.download_file(bucket, key, local_path)
    return local_path

def upload_file_to_s3(local_path, bucket, key):
    s3 = boto3.client("s3")
    logger.info("Uploading %s to s3://%s/%s", local_path, bucket, key)
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"

def train_model(data_path, output_path=None, target_col="target"):
    """
    Trains a simple sklearn pipeline and writes model to output_path.
    - data_path: local CSV path (required)
    - output_path: where to save model.joblib (optional, defaults to models/model.joblib)
    - target_col: name of target column in CSV.
    Returns dict with model metadata.
    """
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    if target_col not in df.columns:
        target_col = df.columns[-1]
        logger.warning("Target column not found; using %s", target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Apply one-hot encoding to match preprocessing
    X = pd.get_dummies(X)

    logger.info("Training model on %d rows, %d features", X.shape[0], X.shape[1])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X, y)

    # Determine output path
    if output_path is None:
        out_dir = os.environ.get("MODEL_OUTPUT_DIR", "models")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "model.joblib")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    joblib.dump(pipe, output_path)
    logger.info("Saved model to %s", output_path)

    # Optionally upload to S3 for deploy step
    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        s3_key = f"models/{os.path.basename(output_path)}"
        s3_uri = upload_file_to_s3(output_path, s3_bucket, s3_key)
        logger.info("Uploaded model to %s", s3_uri)

    return {"model_path": output_path, "n_features": X.shape[1], "n_samples": X.shape[0]}

def main():
    # Keep it robust for CodeBuild
    data_env = os.environ.get("DATA_S3_URI")
    
    local_data = None
    if data_env:
        if data_env.startswith("s3://"):
            local_data = "/tmp/dataset.csv"
            download_s3_to_local(data_env, local_data)
        else:
            local_data = data_env
    else:
        local_example = "data/raw/Topic_15_poc_customers.csv"
        if os.path.exists(local_example):
            local_data = local_example
        else:
            logger.error("No DATA_S3_URI and no local %s; failing", local_example)
            raise SystemExit(1)
    
    train_model(data_path=local_data)

if __name__ == "__main__":
    main()