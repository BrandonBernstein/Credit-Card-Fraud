import pandas as pd
from model_training import model_evaluation, models, param_grid
from data_transformation import date_time_transform, adjust_data, DataTransform
from logger import logging
from exception import ProjectException
from utils import gcp_csv_to_df
import sys
import os

try:

    key = "animated-flare-383117-3112c7a8d011.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(),'key',key)

    train = gcp_csv_to_df('ams561_bucket_creditcardfraud','fraudTrain.csv')
    test = gcp_csv_to_df('ams561_bucket_creditcardfraud','fraudTest.csv')

    date_time_transform(train)
    date_time_transform(test)

    logging.info("Date time transformation complete.")

    adjust_data(train, degree = 3)
    adjust_data(test, degree = 3)

    logging.info("Dropped/Adjusted sensitive data and unneeded columns.")

    columns = list(train.columns[:-1])
    columns.remove("is_fraud")
    X_train = train[columns]
    y_train = train['is_fraud']
    X_test = test[columns]
    y_test = test['is_fraud']

    logging.info("Data has been split.")

    transformer = DataTransform()
    X_train = transformer.transform_train(X_train)
    X_test = transformer.transform_test(X_test)

    logging.info(f"Training data transformation has been completed with shape {X_train.shape}")
    logging.info(f"Testing data transformation has been completed with shape {X_test.shape}")

    logging.info("Model evaluation has begun.")
    results = model_evaluation(X_train,y_train, models, param_grid)
    logging.info("Model evaluation has ended.")

    print(results.items())


except Exception as e:

    logging.info(str(ProjectException(e,sys)))