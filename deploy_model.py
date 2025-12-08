import os
import logging
import boto3
from datetime import datetime
from sagemaker import image_uris

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deploy_model")

def get_sagemaker_image_uri(region, framework="scikit-learn", version="0.23-1"):
    """Get AWS-managed SageMaker container image URI"""
    try:
        return image_uris.retrieve(
            framework=framework,
            region=region,
            version=version,
            py_version="py3",
            instance_type="ml.t3.medium"
        )
    except Exception as e:
        logger.warning("Could not retrieve managed image URI: %s. Using fallback.", e)
        # Fallback to public ECR image
        return f"246618743249.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"

def main():
    bucket = os.environ.get("S3_BUCKET")
    region = os.environ.get("AWS_DEFAULT_REGION")
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
    
    if not all([bucket, region, role_arn]):
        logger.warning("Missing env vars: bucket=%s, region=%s, role=%s", bucket, region, bool(role_arn))
        return
    
    sm = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix="models/")
        if "Contents" not in response or len(response["Contents"]) == 0:
            logger.warning("No models found in S3")
            return
        
        latest_model = max(response["Contents"], key=lambda x: x["LastModified"])
        model_s3_uri = f"s3://{bucket}/{latest_model['Key']}"
        logger.info("Found model: %s", model_s3_uri)
        
        image_uri = get_sagemaker_image_uri(region)
        logger.info("Using image URI: %s", image_uri)
        
        model_name = f"mlops-model-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": image_uri,
                "ModelDataUrl": model_s3_uri
            },
            ExecutionRoleArn=role_arn
        )
        logger.info("Created model: %s", model_name)
        
        config_name = f"mlops-config-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
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
        
        endpoint_name = f"mlops-endpoint-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        logger.info("Created endpoint: %s", endpoint_name)
        
    except Exception as e:
        logger.error("Deployment error: %s", str(e), exc_info=True)

if __name__ == "__main__":
    main()
