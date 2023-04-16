import pandas as pd
from google.cloud import storage
import os
import io
from logger import logging


def gcp_csv_to_df(bucket_name, source_file_name):

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob("Credit Card Data/"+source_file_name)
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))

    logging.info(f'Pulled down file from bucket {bucket_name}, file name: {source_file_name}')

    return df

