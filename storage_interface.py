import os
import json
from pathlib import Path

from google.cloud import storage

credentials_path = str(Path.home() / Path(".gs/mario-geometry-project-0f4ebed1d1e5.json"))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

def upload_blob_from_file(bucket_name: str, source_file_name: str, destination_blob_name: str):
    """
    Uploads a file to the bucket.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

def upload_blob_from_dict(bucket_name: str, data: dict, destination_blob_name: str):
    """
    Uploads a dictionary to the bucket as
    a json file.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    data_json = json.dumps(data)
    blob.upload_from_string(data_json, content_type="application/json")

def read_blob_as_json(bucket_name: str, blob_name: str) -> dict:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    return json.loads(blob.download_as_string())
