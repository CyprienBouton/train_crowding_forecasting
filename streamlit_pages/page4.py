import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import pickle

from src.load_data import load_data
from src.model import predict_model, train_model
from src.preprocess import preprocessing
from src.sncf_dataset import SNCFDataset
from src.custom_buttons import numberNone_input


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
    st.markdown("<h4>Occupancy rate at the same station:</h4>", 
                unsafe_allow_html=True)
    columns_train = st.columns([1,1,1])
    st.markdown("<h4>Occupancy rate of the same train:</h4>", 
                unsafe_allow_html=True)
    columns_station = st.columns([1,1,1])
    for i, name in enumerate(["", "second ", "third "]):
          with columns_train[i]:
                input[f"p{i+1}q0"] = numberNone_input(
                    name + "previous train",
                    label_input=f'p{i+1}q0',
                    min_value=0.0,
                    max_value=1.0)
                
          with columns_station[i]:
                input[f"p0q{i+1}"] = numberNone_input(
                    name + "previous station",
                    label_input=f"p0q{i+1}",
                    min_value=0.0,
                    max_value=1.0)

    file = st.file_uploader("Import your model (.pkl format)",type='pkl')
    # Predict
    if st.button('Predict'):
            if not file:
                st.error(f"🚨 You must first import a model!")
            else:
                model = pickle.load(file)
                input['way'] = 0
                input = preprocessing(pd.DataFrame(input, index=[0]))
                prediction = predict_model(input, model)[0]
                st.markdown(f"The occupancy rate is {round(prediction, 3)}", 
                        unsafe_allow_html=True)
    
