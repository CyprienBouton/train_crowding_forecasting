import streamlit as st
import pickle
import os

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.model import train_model
from src.load_data import load_data

def page5():
    # Set titel
    st.markdown("<h1>Build your model</h1>", unsafe_allow_html=True)
    
    # Get the training dataset and the corresponding labels
    X_train, y_train, _ = load_data(processed=True)
    
    # Build model
    class_models = {
        "Linear regression": LinearRegression,
        "Lasso": Lasso,
        "Ridge": Ridge,
        "SVR": SVR,
        "Decision Tree": DecisionTreeRegressor,
        "Extra Tree": ExtraTreeRegressor,
        "Random Forest": RandomForestRegressor,
    }
    class_model_name = st.selectbox("Type of model", class_models.keys())
    
    if st.button('Train'):
        model_dict = train_model(class_models[class_model_name], X_train, y_train)
        st.write()
        if st.download_button(
                'model', 
                pickle.dumps(model_dict['model']),
                file_name="model.pkl"):
            
            os.remove('models/base_model.pkl')