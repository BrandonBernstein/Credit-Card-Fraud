import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def date_time_transform(data: pd.DataFrame):
    """Splits the trans_date_trans_time column into year, month and day for model preperation.
    Creates new Age variable based off the difference between trans_year and dob.
    Drops dob and trans_date_trans_time for preventing redundancy
    """

    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])

    data['trans_year'] = data['trans_date_trans_time'].dt.year
    data['trans_month'] = data['trans_date_trans_time'].dt.month
    data['trans_week_day'] = data['trans_date_trans_time'].dt.day

    data['dob'] = pd.to_datetime(data['dob'])
    data['age'] = np.round((data['trans_year'] - data['dob'].dt.year), 0)

    data.drop(['trans_date_trans_time', 'dob'], axis=1, inplace=True)


def adjust_data(data: pd.DataFrame, columns: [] = ['first', 'last', 'street', 'cc_num', 'trans_num',
                                                   'merchant', 'city', 'state', 'job', 'zip'],
                adjust: [] = ['lat', 'long'], degree=2):
    """
    Drops sensitive and non sensitive data contained in the data excluding time variables.
    If columns or adjust is empty then drops/adjusts pre assumed variables.

    data: Data Frame to be adjusted
    columns: columns to be dropped
    adjust: columns to be adjusted (meant for longitude and latitude)
    degree: decimal precision to which the latitude and longitude are adjusted too.

    For more information about latitude longitude precision seek
    https://en.wikipedia.org/wiki/Decimal_degrees#Precision.
    """

    data.drop(columns, axis=1, inplace=True)

    data['lat'] = np.round(data['lat'], degree)
    data['long'] = np.round(data['long'], degree)


class DataTransform:

    def __init__(self):
        self.num_col = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
        self.cat_col = ['category', 'gender', 'trans_year', 'trans_month', 'trans_week_day']

    def get_column_transformer(self):
        num_pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ])
        cat_pipe = Pipeline(
            steps=[
                ("one_hot", OneHotEncoder(handle_unknown = 'ignore'))
            ])

        column_transformer = ColumnTransformer([
            ('numerical', num_pipe, self.num_col),
            ('categorical', cat_pipe, self.cat_col)
        ])

        return column_transformer

    def transform_train(self, data: pd.DataFrame):
        self.preprocess = self.get_column_transformer()

        data = self.preprocess.fit_transform(data)

        return data

    def transform_test(self, data: pd.DataFrame):
        data = self.preprocess.transform(data)

        return data
