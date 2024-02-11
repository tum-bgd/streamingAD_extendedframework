import argparse
import json
import os
import sys
import logging
import subprocess
import boto3

# subprocess.run(['add-apt-repository', 'ppa:deadsnakes/ppa', '-y'])
# subprocess.run(['apt-get', '-y', 'install', 'python3.8'])
# subprocess.run(['python3.8', '-m', 'venv', 'streamingADenv'])
# subprocess.run(['source', 'streamingADenv/bin/activate'])
# subprocess.run(['pip', 'install', 'numpy'])
# subprocess.run(['pip', 'install', 'cython==0.29.14'])
# subprocess.run(['pip', 'install', 'git+https://github.com/andreasckoch/eif.git'])
subprocess.run(['pip', 'install', 'eif==1.0.2'])
subprocess.run(['pip', 'install', 'nbeats-keras'])
subprocess.run(['pip', 'install', 'statsmodels'])
subprocess.run(['pip', 'install', 'scipy'])

from main import main
from utils_aws import upload_folder_to_s3

# ------------------------------------------------------------------------------
# Set logger
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# ------------------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------------------
def parse_args():
    # Initialization
    parser = argparse.ArgumentParser()

    # Static hyperparameters
    parser.add_argument("--tuning_job_name", type=str, default="default_tuning_job_name")
    parser.add_argument("--s3_output_dir", type=str, default="andreas/workspace/output")
    parser.add_argument("--s3_tuning_job_file_path", type=str, default="s3://aiteamads/andreas/workspace/streamingAD_extendedframework/tuning_job/default.json")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--training_set_length", type=int, default=5000)
    parser.add_argument("--data_representation_length", type=int, default=100)
    parser.add_argument("--dataset_category", type=str, default="univariate")
    parser.add_argument("--collection_id", type=str, default="KDD-TSAD")

    # Dynamic hyperparameters
    parser.add_argument("--irrelevant_parameter1", type=int, default=1)

    # Environment variables given by the training image
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_DATA_DIR"])
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--training_env", type=json.loads, default=os.environ["SM_TRAINING_ENV"])

    args = parser.parse_args()    
    return args

if __name__ == "__main__":
    args = parse_args()

    logger.info("---Just for CloudWatch---   Validation_Loss=0.000;")
    
    # choose dataset_id
    bucket_name = 'aiteamads'
    s3_resource = boto3.resource('s3')
    s3_resource.meta.client.download_file(bucket_name, args.s3_tuning_job_file_path, 'dataset_ids.json')
    with open('dataset_ids.json', 'r') as file:
        dataset_ids = json.load(file)
    dataset_id = dataset_ids[0]
    with open('dataset_ids.json', 'w') as file:
        json.dump(dataset_ids[1:], file)
    s3_resource.meta.client.upload_file('dataset_ids.json', bucket_name, args.s3_tuning_job_file_path)
    print(f"--- DATASET ID: {dataset_id} ---")

    local_output_dir = f'{dataset_id}/{args.tuning_job_name}'
    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir)
    
    main(
        args.data_dir,
        local_output_dir,
        args.dataset_category, 
        args.collection_id, 
        dataset_id, 
        args.batch_size, 
        args.training_set_length, 
        args.data_representation_length
    )
    
    # upload to s3
    upload_folder_to_s3(bucket_name, local_output_dir, args.s3_output_dir)
    