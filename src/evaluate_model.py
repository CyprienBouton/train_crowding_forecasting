from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from .sncf_dataset import SNCFDataset


def evaluate_model(class_model, dataset, labels, model_params={},
                   n_splits=5, random_state=None, method_context='median',
                   method_lags='smart'):
  """ Evaluate the performance of a model on the dataset.

  :param class_model: Class used to generate the model.
  :type class_model: Class to generate a ml_model
  :param dataset: SNCF Dataset.
  :type dataset: pd.DataFrame
  :param labels: Occupancy rates of the trains
  :type labels: pd.Series
  :param model_params: Parameters used to generate the model.
  :param type: dict
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
  with tqdm(total=n_splits) as pbar:
    for train_idx, _ in k_fold.split(dataset):
      # Split dataset
      X_train, X_val = dataset.iloc[train_idx], dataset.iloc[~train_idx]
      y_train, y_val = labels.iloc[train_idx], labels.iloc[~train_idx]

      # Transform to Dataset objects
      X_train = SNCFDataset(X_train)
      X_val = SNCFDataset(X_val)

      # Replace nan values
      impute_missing_dict = X_train.train_impute_missing_values(
        method_context=method_context, method_lags=method_lags)
      X_val.predict_impute_missing_values(impute_missing_dict)

      # Scale data
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_val = scaler.transform(X_val)
      # Initiate model
      model = class_model(**model_params)
      # Training
      model.fit(X_train, y_train.values.ravel())
      # Prediction
      preds = model.predict(X_val)
      # Append regression score
      scores.append(mean_squared_error(y_val, preds))

      # Display progress bar
      pbar.update(1)

  return sum(scores)/n_splits