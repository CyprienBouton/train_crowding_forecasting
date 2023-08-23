import pandas as pd
import datetime
from src.sncf_dataset import SNCFDataset


def datetime_to_float(d):
    # Function to convert datetime to float
    epoch = datetime.datetime.utcfromtimestamp(0)
    total_seconds =  (d - epoch).total_seconds()
    return total_seconds


def get_train_columns():
    # Get all columns of the train dataset
    X_train = pd.read_pickle('datasets/X_train')
    return X_train.columns


def preprocessing(dataset, inference=True):
    """ Convert non numerical values to numerical values

    :param dataset: dataset.
    :type datset: pandas.dataframe
    :param inference: Whether the processing is done during training or inference.
    :type inference: bool
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

    # Delete constant column
    dataset = dataset.drop('way', axis=1)

    # Vectorize train and station columns
    dataset.train = dataset.train.astype(str) # convert train id to string
    dataset = pd.get_dummies(dataset, dtype=float)
    # Add missing columns during inference
    if inference:
        # Get missing columns
        missing_cols = set(get_train_columns()) - set(dataset.columns)
        for c in missing_cols:
            dataset[c] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        dataset = dataset[get_train_columns()]

    # Convert composition to float
    dataset.composition = dataset.composition.astype(float)
    # return a SNCFDataset
    return dataset