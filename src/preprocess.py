import pandas as pd
import datetime
from src.sncf_dataset import SNCFDataset


def datetime_to_float(d):
    # Function to convert datetime to float
    epoch = datetime.datetime.utcfromtimestamp(0)
    total_seconds =  (d - epoch).total_seconds()
    return total_seconds


def preprocessing(dataset):
    """ Convert non numerical values to numerical values

    :param dataset: dataset.
    :type datset: pandas.dataframe
    :return dataset: Preprocessed dataset.
    :rtype dataset: SNCFDataset
    """
    # convert dates to datetime
    dataset.date = pd.to_datetime(dataset.date)

    # Add weekday
    dataset['weekday'] = dataset.date.apply(lambda x: int(x.weekday())).astype(float)
    # convert datetime to float
    dataset.date = dataset.date.apply(datetime_to_float)

    # Convert hour to float
    dataset.hour = pd.to_timedelta(dataset.hour).dt.components['hours']

    # Vectorize train and station columns
    dataset.train = dataset.train.astype(str) # convert train id to string
    dataset = pd.get_dummies(dataset, dtype=float)

    # Delete constant column
    dataset.drop('way', axis=1, inplace=True)

    # Convert composition to float
    dataset.composition = dataset.composition.astype(float)
    # return a SNCFDataset
    return SNCFDataset(dataset)