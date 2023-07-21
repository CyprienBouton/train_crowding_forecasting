from sklearn.linear_model import LinearRegression
from src.evaluate_model import evaluate_model
from src.load_data import load_data
from src.preprocess import preprocessing

X, y, _ = load_data()
preprocessing(X)