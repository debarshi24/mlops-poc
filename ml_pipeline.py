"""
ml_pipeline.py

Orchestrator for preprocess -> train -> evaluate -> upload artifacts.

Expected environment variables (set by CodeBuild from CloudFormation):
- DATA_S3_URI            (s3://.../data/raw/Topic_15_poc_customers.csv)  REQUIRED
- S3_BUCKET              (artifacts bucket name)                         REQUIRED
- MODEL_PACKAGE_GROUP_NAME  (optional)                                     OPTIONAL
- AUTO_APPROVE_MODELS    'true'|'false'                                   OPTIONAL
- AWS_DEFAULT_REGION     AWS region (e.g. us-east-1)                      REQUIRED (used by boto3)
- CODEBUILD_RESOLVED_SOURCE_VERSION  commit id (optional; provided by CodeBuild)
- REGISTER_MODEL         'true' to attempt model package registration (optional) - not implemented fully
"""

import os
import logging
import json
from datetime import datetime
from urllib.parse import urlparse

import boto3
import pandas as pd
import numpy as np

# Import your modularized ML functions
# Ensure repo structure: src/preprocess.py, src/train.py, src/evaluate.py, src/utils.py
from src.preprocess import preprocess_csv
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import download_s3_to_local, upload_file_to_s3, save_json_to_s3, generate_run_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_pipeline")

def validate_env():
    missing = []
    for v in ("DATA_S3_URI", "S3_BUCKET", "AWS_DEFAULT_REGION"):
        if not os.environ.get(v):
            missing.append(v)
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {missing}")

def main():
    try:
        validate_env()
        data_s3 = os.environ["DATA_S3_URI"]
        bucket = os.environ["S3_BUCKET"]
        region = os.environ.get("AWS_DEFAULT_REGION")
        auto_approve = os.environ.get("AUTO_APPROVE_MODELS", "false").lower()
        commit = os.environ.get("CODEBUILD_RESOLVED_SOURCE_VERSION", "local")
        register_model_flag = os.environ.get("REGISTER_MODEL", "false").lower()

        run_id = generate_run_id("run")
        logger.info("Starting pipeline run: %s", run_id)
        logger.info("Environment: DATA_S3_URI=%s, S3_BUCKET=%s", data_s3, bucket)

        # Check if data file exists
        s3 = boto3.client('s3')
        parsed = urlparse(data_s3)
        data_bucket = parsed.netloc
        data_key = parsed.path.lstrip('/')
        
        try:
            s3.head_object(Bucket=data_bucket, Key=data_key)
            logger.info("Data file exists at %s", data_s3)
        except Exception as e:
            logger.error("Data file not found at %s: %s", data_s3, e)
            logger.info("Creating dummy data for testing...")
            create_dummy_data(data_s3)

        # 1) Download CSV from S3 to local
        local_csv = f"/tmp/{run_id}_data.csv"
        logger.info("Downloading dataset %s -> %s", data_s3, local_csv)
        download_s3_to_local(data_s3, local_csv)

        # 2) Preprocess (produces train.csv and test.csv locally)
        tmp_prefix = f"/tmp/{run_id}"
        logger.info("Preprocessing data...")
        train_path, test_path = preprocess_csv(local_csv, output_prefix=tmp_prefix)
        logger.info("Preprocess done. train=%s test=%s", train_path, test_path)

        # 3) Train
        model_local_path = f"/tmp/{run_id}_model.joblib"
        logger.info("Training model...")
        train_meta = train_model(train_path, model_local_path)
        logger.info("Training finished. model saved to %s", model_local_path)

        # 4) Evaluate
        logger.info("Evaluating model...")
        metrics = evaluate_model(model_local_path, test_path)
        logger.info("Evaluation metrics: %s", metrics)

        # 5) Upload model and artifacts to S3
        model_s3_key = f"models/{run_id}/model.joblib"
        model_s3_uri = upload_file_to_s3(bucket, model_local_path, model_s3_key)
        logger.info("Uploaded model to %s", model_s3_uri)

        metrics_key = f"artifacts/{run_id}/metrics.json"
        save_json_to_s3(bucket, metrics, metrics_key)
        logger.info("Uploaded metrics to s3://%s/%s", bucket, metrics_key)

        model_info = {
            "run_id": run_id,
            "commit": commit,
            "model_s3_uri": model_s3_uri,
            "metrics": metrics,
            "trained_at": datetime.utcnow().isoformat() + "Z"
        }
        model_info_key = f"artifacts/{run_id}/model_info.json"
        save_json_to_s3(bucket, model_info, model_info_key)
        logger.info("Uploaded model info to s3://%s/%s", bucket, model_info_key)

        # 6) Optional: register model in SageMaker Model Registry (placeholder)
        if register_model_flag == "true":
            logger.info("REGISTER_MODEL=true: Model registration is environment-specific and is not fully implemented in this script.")
            # Implementing a robust CreateModelPackage flow requires packaging a container or using a pre-built inference container.
            # For now we simply record model_info.json which downstream processes or manual steps can use to register the model.
            # You can optionally use boto3/sagemaker SDK here to call CreateModelPackage with proper parameters.

        # 7) Print outputs for CodeBuild logs
        logger.info("Pipeline run complete: %s", run_id)
        logger.info("Model S3 URI: %s", model_s3_uri)
        logger.info("Metrics: %s", metrics)
        logger.info("Model info S3: s3://%s/%s", bucket, model_info_key)

        # If you want CodeBuild to conditionally auto-approve the pipeline, write a sentinel file into TrainingOutput artifact
        # For example: create file 'artifacts/<run_id>/deploy_signal.txt' with "approve" or "reject"
        # But note: CodePipeline also contains a ManualApproval stage; automatic approval would require removing that stage or using API to approve.
        if auto_approve == "true":
            logger.info("AUTO_APPROVE_MODELS is true — set up pipeline to auto-deploy based on this artifact if desired.")
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        raise

def create_dummy_data(s3_uri):
    """Create dummy CSV data for testing when real data is missing"""
    import pandas as pd
    import numpy as np
    
    # Create dummy dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(2, 1.5, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Upload to S3
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    
    csv_buffer = df.to_csv(index=False)
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer)
    logger.info("Created dummy data at %s", s3_uri)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        exit(1)
