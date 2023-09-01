import numpy as np
from stqdm import stqdm
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from .sncf_dataset import SNCFDataset


def train_model(class_model, dataset, labels, model_params={}, 
                method_context='median', method_lags='smart'):
    """ Train a model to predict train occupancy rate.

    :param class_model: Class used to generate a model.
    :type class_model: Class to generate a ml_model
    :param dataset: train dataset.
    :type dataset: pd.DataFrame
    :param labels: Occupancy rates of the trains
    :type labels: pd.Series
    :param model_params: Parameters used to generate the model.
    :type model_params: dict
    :param method_context: Method used to impute missing values of context columns.
    :type method_context: str
    :param method_lags: Method used to impute missing values of lags columns.
    :type method_lags: str
    :return models_dict: Dictionary containing the trained models.
    :rtype models_dict: dict
    """
    # Convert dataset to SNCF Dataset
    dataset = SNCFDataset(dataset)

    # Replace nan values
    impute_missing_dict = dataset.train_impute_missing_values(
                method_context=method_context, method_lags=method_lags)
    
    # Scale data
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    # Initiate model
    model = class_model(**model_params)
    # Training
    model.fit(dataset, labels.values.ravel())
    
    models_dict = {
        "impute_missing_dict": impute_missing_dict,
        "scaler": scaler,
        "model": model,
    }
    return models_dict


def predict_model(dataset, models_input):
    """ Predict train occupancy rate.

    :param dataset: train dataset.
    :type dataset: pd.DataFrame
    :param models_input: Trained models.
    :type models_input: dict or ml model
    :return occupancy_rates: Predicted occupancy rates within the trains.
    :rtype occupancy_rates: np.array
    """
    # If only one model is provided then load other models
    if not isinstance(models_input, dict):
        models_dict = {}
        models_dict['model'] = models_input
        for missing_model_name in ['impute_missing_dict', 'scaler']:
            models_dict[missing_model_name] = pickle.load(
                open('models/'+missing_model_name, 'rb'))
    else:
        models_dict = models_input
    
    # Convert dataset to SNCF Dataset
    dataset = SNCFDataset(dataset)

    # Replace nan values
    dataset.predict_impute_missing_values(models_dict['impute_missing_dict'])
    
    # Scale data
    dataset = models_dict['scaler'].transform(dataset)
    # Predict
    occupancy_rate = models_dict['model'].predict(dataset)
    # Constraints values from 0 to 1
    occupancy_rates = np.clip(occupancy_rate, 0, 1)
    
    return occupancy_rates


def evaluate_model(class_model, dataset, labels, model_params={},
                   n_splits=5, random_state=None, method_context='median',
                   method_lags='smart'):
    """ Evaluate the performance of a model on the dataset.

    :param class_model: Class used a generate the model.
    :type class_model: Class to generate a ml_model
    :param dataset: train Dataset.
    :type dataset: pd.DataFrame
    :param labels: Occupancy rates of the trains
    :type labels: pd.Series
    :param model_params: Parameters used to generate the model.
    :type model_params: dict
    :param n_splits: Number of folds.
    :type n_splits: int
    :param random_state: Control the randomness of the shuffle.
    :type random_state: None or int
    :param method_context: Method used to impute missing values of context columns.
    :type method_context: str
    :param method_lags: Method used to impute missing values of lags columns.
    :type method_lags: str
    """

    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    with stqdm(total=n_splits) as pbar:
        for train_idx, _ in k_fold.split(dataset):
            # Split dataset
            X_train, X_val = dataset.iloc[train_idx], dataset.iloc[~train_idx]
            y_train, y_val = labels.iloc[train_idx], labels.iloc[~train_idx]
            
            # Train models
            models_dict = train_model(class_model, X_train, y_train, model_params, 
                        method_context, method_lags)
            # Predict occupancy rates
            preds = predict_model(X_val, models_dict)
            # Append regression score
            scores.append(r2_score(y_val, preds))
            # Display progress bar
            pbar.update(1)

    return sum(scores)/n_splits