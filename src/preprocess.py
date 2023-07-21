import pandas as pd
import datetime


def datetime_to_float(d):
    # Function to convert datetime to float
    epoch = datetime.datetime.utcfromtimestamp(0)
    total_seconds =  (d - epoch).total_seconds()
    return total_seconds


def vectorize_columns(dataset, columns):
  """ Vectorize the given columns.

  :param dataset: Dataset on which transformation is applied
  :type dataset: pandas.DataFrame
  :param columns: Columns to vectorized.
  :type columns: list of str or str
  """
  if isinstance(columns, str):
    columns = [columns]
  assert isinstance(columns, list)

  # Looping on the columns to vectorized
  for column in columns:

    # Looping on the different values
    for i, value in enumerate(dataset[column].unique()):
      dataset[column+'_'+str(i+1)] = (dataset[column]==value).astype(float)

  # Delete column
  dataset.drop(columns, axis=1, inplace=True)


def preprocessing(dataset):
  """ Convert non numerical values to numerical values

  :param dataset: dataset.
  :type datset: pandas.dataframe
  """
  # convert dates to datetime
  dataset.date = pd.to_datetime(dataset.date)

  # Add weekday
  dataset['weekday'] = dataset.date.apply(lambda x: int(x.weekday())).astype(float)
  # convert datetime to float
  dataset.date = dataset.date.apply(datetime_to_float)

  # Convert hour to float
  dataset.hour = pd.to_timedelta(dataset.hour).dt.components['hours']

  # Vectorize station and train columns
  dataset.train = dataset.train.astype(str) # First convert data to string
  vectorize_columns(dataset, ['station', 'train'])

  # Delete constant column
  dataset.drop('way', axis=1, inplace=True)

  # Convert composition to float
  dataset.composition = dataset.composition.astype(float)
