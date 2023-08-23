import pandas as pd
import pickle

def load_data(processed=False):
    """Load all datasets: X, y and X_test

    :param processed: Wether to return processed datasets
    :return X: train dataset
    :rtype X: pd.DataFrame
    :return y: labels
    :rtype y: pd.DataFrame
    :return X_test: test dataset
    :rtype X_test: pd.DataFrame
    """
    if processed:
        X_train = pickle.load(open('datasets/X_train', 'rb'))
        y_train = pickle.load(open('datasets/y_train', 'rb'))
        X_test = pickle.load(open('datasets/X_test', 'rb'))
    
    else:
        X_train = pd.read_csv('datasets/Xtrain_hgcGIrA.csv')
        y_train = pd.read_csv('datasets/Ytrain_yL5OjS4.csv')
        X_test = pd.read_csv('datasets/Xtest.csv')
        # Get rid of the row number
        y_train.drop(columns='Unnamed: 0', inplace=True)
    
    return X_train, y_train, X_test

