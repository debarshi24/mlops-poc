import os
import logging
import boto3
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deploy_model")

def main():
    bucket = os.environ.get("S3_BUCKET")
    region = os.environ.get("AWS_DEFAULT_REGION")
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
    
    if not all([bucket, region, role_arn]):
        logger.info("Missing required env vars for deployment")
        return
    
    sm = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    
    # Find latest model in S3
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix="models/")
        if "Contents" not in response:
            logger.info("No models found in S3")
            return
        
        latest_model = max(response["Contents"], key=lambda x: x["LastModified"])
        model_s3_uri = f"s3://{bucket}/{latest_model['Key']}"
        logger.info("Found model: %s", model_s3_uri)
    except Exception as e:
        logger.error("Error finding model: %s", e)
        return
    
    # Create model
    model_name = f"mlops-model-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    try:
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": "246618743249.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
                "ModelDataUrl": model_s3_uri,
                "Environment": {"SAGEMAKER_PROGRAM": "inference.py", "SAGEMAKER_SOURCE_DIR": "s3://bucket/src/model"}
            },
            ExecutionRoleArn=role_arn
        )
        logger.info("Created model: %s", model_name)
    except Exception as e:
        logger.error("Error creating model: %s", e)
        return
    
    # Create endpoint config
    config_name = f"mlops-config-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    try:
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                "VariantName": "Primary",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.t3.medium"
            }]
        )
        logger.info("Created endpoint config: %s", config_name)
    except Exception as e:
        logger.error("Error creating endpoint config: %s", e)
        return
    
    # Create endpoint
    endpoint_name = f"mlops-endpoint-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    try:
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        logger.info("Created endpoint: %s", endpoint_name)
    except Exception as e:
        logger.error("Error creating endpoint: %s", e)
        return

if __name__ == "__main__":
    main()
