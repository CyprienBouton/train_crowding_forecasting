import streamlit as st

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import ExtraTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def page4():

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