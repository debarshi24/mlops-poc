import os
import boto3
from urllib.parse import urlparse
import json
from datetime import datetime




def parse_s3_uri(s3_uri):
parsed = urlparse(s3_uri)
if parsed.scheme != 's3':
raise ValueError('s3_uri must start with s3://')
bucket = parsed.netloc
key = parsed.path.lstrip('/')
return bucket, key




def download_s3_to_local(s3_uri, local_path=None):
bucket, key = parse_s3_uri(s3_uri)
if local_path is None:
local_path = '/tmp/' + key.split('/')[-1]
s3 = boto3.client('s3')
os.makedirs(os.path.dirname(local_path), exist_ok=True)
s3.download_file(bucket, key, local_path)
return local_path




def upload_file_to_s3(bucket, local_path, key):
s3 = boto3.client('s3')
s3.upload_file(local_path, bucket, key)
return f"s3://{bucket}/{key}"




def save_json_to_s3(bucket, data, key):
s3 = boto3.client('s3')
s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(data).encode('utf-8'))
return f"s3://{bucket}/{key}"




def generate_run_id(prefix='run'):
return prefix + '-' + datetime.utcnow().strftime('%Y%m%d%H%M%S')




def list_s3_prefix(bucket, prefix):
s3 = boto3.client('s3')
paginator = s3.get_paginator('list_objects_v2')
keys = []
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
for obj in page.get('Contents', []):
keys.append(obj['Key'])
return keys