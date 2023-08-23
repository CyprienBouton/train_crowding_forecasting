import numpy as np
import pandas as pd
import warnings

from sklearn.linear_model import LinearRegression


class SNCFDataset(pd.DataFrame):
    # Hide dataframe warning that rose when creating new attributes
    warnings.filterwarnings("ignore", message="Pandas doesn't allow columns to be created via a new attribute name")

    def __init__(self, dataset):
        super().__init__(dataset)
        self.trains_columns = dataset.columns.values[
            np.array([c[:5]=='train' for c in dataset.columns.values])]
        self.stations_columns = dataset.columns.values[
            np.array([c[:7]=='station' for c in dataset.columns.values])]
        self.context_columns = dataset.columns.values[
            ~np.array([c[:1]=='p' for c in dataset.columns.values])]

    # Auto-updating attribute
    @property
    def limited_cols(self):
        # Dataset without trains and stations columns
        return self.drop(np.concatenate([self.trains_columns,
                                        self.stations_columns]), axis=1)
    @property
    def trains_data(self):
        return self[self.trains_columns]

    @property
    def stations_data(self):
        return self[self.stations_columns]

    def replace_nan(self, column, value):
        """ Replace nan values of a given column.

        :param column: Name of the column to be transformed.
        :type column: str
        :param value: Value used to replace nan data.
        :type value: str or pd.Series
        """
        # If value is a string
        if isinstance(value, str):
            if value == 'median':
                new_column = self[column].median()
            elif value == 'mode':
                new_column = self[column].mode()[0]
            # Check if value is a column
            elif value in self.columns:
                new_column = self.loc[self[column].isnull(), value]
            else:
                raise NotImplementedError(f'{value} cannot be used to replace {column}.')
        else:
            new_column = value
        # Replace nan values with the new column
        self.loc[self[column].isnull(), column] = new_column

    def train_impute_missing_values(self,
                                method_context='median',
                                method_lags='smart'):
        """ Impute all missing values during training.
        :param method_context: Method used to impute contextual variables.
        :type method_contex: str
        :param method_lags: Method used to impute lags variables.
        :type method_lags: str
        :return impute_missing_dict: Dictionary of paramters to impute missing data
        :rtype impute_missing_dict: dict
        """
        impute_missing_dict = {
            'method_lags': method_lags,
        }
        ### Impute hour column
        if method_context == 'median':
            self.replace_nan('hour', 'median')
            impute_missing_dict['hour'] = self.hour.median()
        elif method_context == 'mode':
            self.replace_nan('hour', 'mode')
            impute_missing_dict['hour'] = self.hour.mode()[0]
        else:
            raise NotImplementedError(f"The method '{method_context}'" \
                                    "hasn't been implemented.")
        ### Impute lags columns
        if method_lags == 'median':
            for column in ['p1q0', 'p2q0', 'p3q0', 'p0q1', 'p0q2', 'p0q3']:
                self.replace_nan(column, 'median')
                impute_missing_dict[column] = self[column].median()

        elif method_lags == 'mode':
            for column in ['p1q0', 'p2q0', 'p3q0', 'p0q1', 'p0q2', 'p0q3']:
                self.replace_nan(column, 'mode')
                impute_missing_dict[column] = self[column].mode()[0]

        elif method_lags == 'smart':
            #Set p1q0 and p0q1
            for column in ['p1q0', 'p0q1']:
                self.replace_nan(column, 'median')
                impute_missing_dict[column] = self[column].median()

            # Replace the second previous occupancy rates by the previous ones
            for column, value in zip(['p2q0', 'p0q2'], ['p1q0', 'p0q1']):
                self.replace_nan(column, value)
            impute_missing_dict = self._train_p3q0_p0q3(impute_missing_dict)


        else:
            raise NotImplementedError(f"The method '{method_lags}'"
                                      +" hasn't been implemented.")

        return impute_missing_dict

    def predict_impute_missing_values(self, impute_missing_dict):
        """ Impute training values during inference.
        :param impute_missing_dict: Dictionary of paramters to impute missing data.
        :type impute_missing_dict: dict
        impute_missing_dict have the following keys:
        - 'method_lags'
        - 'hour'
        - 'p1q0'
        - 'p0q1'
        - 'p2q0', 'p3q0', 'p0q2', 'p0q3' if method_lags != 'smart'
        - 'model_p3q0', 'model_p0q3' if method_lags == 'smart'
        """
        ### Hour column
        self.replace_nan('hour', impute_missing_dict['hour'])

        ### Lags columns
        if impute_missing_dict['method_lags'] != 'smart':
            for columm in ['p1q0', 'p2q0', 'p3q0', 'p0q1', 'p0q2', 'p0q3']:
                self.replace_nan(column, impute_missing_dict[column])
        else:
            # Replace the second previous occupancy rates by the previous ones
            for column in ['p1q0', 'p0q1']:
                self.replace_nan(column, impute_missing_dict[column])
            for column, value in zip(['p2q0', 'p0q2'], ['p1q0', 'p0q1']):
                self.replace_nan(column, value)
            self._predict_p3q0_p0q3(impute_missing_dict)

    def _predict_p3q0_p0q3(self, impute_missing_dict):
        # Predict the occupancy rates of the third previous train
        if len(self[self.p3q0.isnull()])>0:
            model_p3q0 = impute_missing_dict['model_p3q0']
            null_data = self.p3q0.isnull()
            input = list(zip(self[null_data].p1q0, self[null_data].p2q0))
            self.replace_nan('p3q0', model_p3q0.predict(input))
        # Predict the occupancy rates at the third previous station
        if len(self[self.p0q3.isnull()])>0:
            model_p0q3 = impute_missing_dict['model_p0q3']
            null_data = self.p0q3.isnull()
            input = list(zip(self[null_data].p0q1, self[null_data].p0q2))
            self.replace_nan('p0q3', model_p0q3.predict(input))

    def _train_p3q0_p0q3(self, impute_missing_dict):
            # Predict the occupancy rates of the third previous train
        if len(self[self.p3q0.isnull()])>0:
            model_p3q0 = LinearRegression()
            null_data = self.p3q0.isnull()
            # Training
            input = list(zip(self[~null_data].p1q0, self[~null_data].p2q0))
            model_p3q0.fit(input, self[~null_data].p3q0)
            # Prediction
            input = list(zip(self[null_data].p1q0, self[null_data].p2q0))
            self.replace_nan('p3q0', model_p3q0.predict(input))
            impute_missing_dict['model_p3q0'] = model_p3q0

        # Predict the occupancy rates at the third previous station
        if len(self[self.p0q3.isnull()])>0:
            model_p0q3 = LinearRegression()
            null_data = self.p0q3.isnull()
            # Training
            input = list(zip(self[~null_data].p0q1, self[~null_data].p0q2))
            model_p0q3.fit(input, self[~null_data].p0q3)
            # Prediction
            input = list(zip(self[null_data].p0q1, self[null_data].p0q2))
            self.replace_nan('p0q3', model_p0q3.predict(input))
            impute_missing_dict['model_p0q3'] = model_p0q3
        return impute_missing_dict