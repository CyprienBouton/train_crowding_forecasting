import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import pickle

from src.load_data import load_data
from src.model import predict_model, train_model
from src.preprocess import preprocessing
from src.sncf_dataset import SNCFDataset

def page4():
    # Set title
    st.markdown("<h1>Occupancy rate forecasting</h1>", unsafe_allow_html=True)
    
    # Get train and station columns
    X_train, _, _ = load_data(processed=True)
    X_train = SNCFDataset(X_train)   
    train_columns, station_columns = X_train.trains_columns, X_train.stations_columns
    input = {}
    
    # Contextual variables
    input['date'] = st.date_input('Date of train passage', 
                  value= datetime(2019,1,7), 
                  min_value=datetime(2019,1,7), 
                  max_value=datetime(2019,5,20), 
                  help="Date must be a weekday between January 7th and May 20th")
    input['date'] = input['date'].strftime('%Y-%m-%d')
    
    input['train'] = st.selectbox("Train id",
                              train_columns)
    
    input['station'] = st.selectbox("Station id",
                           station_columns)
    
    input['hour'] = st.time_input('Time slot')
    input['hour'] = input['hour'].strftime("%H:%M:%S")

    input['composition'] = st.selectbox('Number of train units', [1, 2])
    
    # Lags variables
    lag_variable_used = {}
    for lag_variable in ['p1q0', 'p2q0', 'p3q0',
                         'p0q1', 'p0q2', 'p0q3']:
            
            lag_variable_used[lag_variable] = st.checkbox(lag_variable + ' exist')
            
            input[lag_variable] = st.number_input(
                lag_variable,
                min_value=0.0,
                max_value=1.0,
                label_visibility="collapsed", 
                disabled=not lag_variable_used[lag_variable])
            
            if not lag_variable_used[lag_variable]:
                input[lag_variable] = np.nan

    file = st.file_uploader("Import your model (.pkl format)",type='pkl')
    model = pickle.load(file)
    # Predict
    if st.button('Predict'):
            input['way'] = 0
            input = preprocessing(pd.DataFrame(input, index=[0]))
            st.write(str(predict_model(input, model)[0]))
    
