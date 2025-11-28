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

def train_model(data_path=None, target_col="target"):
    """
    Trains a simple sklearn pipeline and writes models/model.joblib.
    - data_path: local CSV path or s3://... path. If None, reads DATA_S3_URI env var.
    - target_col: name of target column in CSV.
    Returns path to local model file.
    """
    # Resolve dataset
    if data_path is None:
        data_path = os.environ.get("DATA_S3_URI", "data/raw/Topic_15_poc_customers.csv")

    local_data = data_path
    if data_path.startswith("s3://"):
        local_data = "/tmp/dataset.csv"
        download_s3_to_local(data_path, local_data)

    logger.info("Loading data from %s", local_data)
    df = pd.read_csv(local_data)

    if target_col not in df.columns:
        # try to find last column as target if default missing
        target_col = df.columns[-1]
        logger.warning("Target column not found; using %s", target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info("Training model on %d rows, %d features", X.shape[0], X.shape[1])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X, y)

    # Ensure output dir
    out_dir = os.environ.get("MODEL_OUTPUT_DIR", "models")
    os.makedirs(out_dir, exist_ok=True)
    local_model_path = os.path.join(out_dir, "model.joblib")
    joblib.dump(pipe, local_model_path)
    logger.info("Saved model to %s", local_model_path)

    # Optionally upload to S3 for deploy step
    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        s3_key = f"models/{os.path.basename(local_model_path)}"
        s3_uri = upload_file_to_s3(local_model_path, s3_bucket, s3_key)
        logger.info("Uploaded model to %s", s3_uri)

    return local_model_path

def main():
    # Keep it robust for CodeBuild
    data_env = os.environ.get("DATA_S3_URI")
    if data_env:
        train_model(data_path=data_env)
    else:
        # if local data exists, use it; otherwise fail with helpful log
        local_example = "data/raw/Topic_15_poc_customers.csv"
        if os.path.exists(local_example):
            train_model(data_path=local_example)
        else:
            logger.error("No DATA_S3_URI and no local %s; failing", local_example)
            raise SystemExit(1)

if __name__ == "__main__":
    main()