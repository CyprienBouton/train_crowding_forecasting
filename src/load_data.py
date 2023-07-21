import pandas as pd

def load_data():
  """Load all datasets: X, y and X_test

  :param X: train dataset
  :type X: pd.DataFrame
  :param y: labels
  :type y: pd.DataFrame
  :param X_test: test dataset
  :type X_test: pd.DataFrame
  """
  X = pd.read_csv('datasets/Xtrain_hgcGIrA.csv')
  y = pd.read_csv('datasets/Ytrain_yL5OjS4.csv')
  X_test = pd.read_csv('datasets/Xtest.csv')

  # Get rid of the row number
  y.drop(columns='Unnamed: 0', inplace=True)
  return X, y, X_test

