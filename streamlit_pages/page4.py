import streamlit as st
from datetime import datetime

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import ExtraTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from src.load_data import load_data

def page4():

    X_train, _, _ = load_data()

    class_model_names = {
        'Linear Model': LinearRegression,
        'Lasso': Lasso,
        'Decision Tree': DecisionTreeClassifier,
        'Extra Tree Regressor': ExtraTreeRegressor,
        'Random Forest': RandomForestRegressor,
        'SVR': SVR
    }

    st.markdown("<h1>Occupancy rate forecasting</h1>", unsafe_allow_html=True)
    model_used = st.selectbox("What model would you like to use?",
                              class_model_names.keys())
    
    # Contextual variables
    date = st.date_input('Date of train passage', 
                  value= datetime(2019,1,7), 
                  min_value=datetime(2019,1,7), 
                  max_value=datetime(2019,5,20), 
                  help="Date must be a weekday between January 7th and May 20th")
    
    train_id = st.selectbox("Train id",
                              sorted(X_train.train.unique()))
    
    station = st.selectbox("Station id",
                           sorted(X_train.station.unique()))
    
    hour = st.time_input('Time slot')

    composition = st.selectbox('Number of train units', [1, 2])
    # Lags variables


