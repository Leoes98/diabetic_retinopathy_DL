import pandas as pd

from TaxiFareModel.utils import simple_time_tracker
from google.cloud import storage
from TaxiFareModel.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH


@simple_time_tracker
def get_data_from_gcp(nrows=10000, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
    data = path
    return data




if __name__ == '__main__':
    df = get_data()
