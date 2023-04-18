import pandas as pd
from google.cloud import storage
import io
import pickle
from logger import logging


def gcp_csv_to_df(bucket_name, source_file_name):

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob("Credit Card Data/"+source_file_name)
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))

    logging.info(f'Pulled down file from bucket {bucket_name}, file name: {source_file_name}')

    return df

def save_pickle(object, file_name):

    with open(f"{file_name}.pickle", "wb") as save:
        pickle.dump(object, save)

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket.- From GC Documentation"""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
