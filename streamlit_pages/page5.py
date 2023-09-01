import streamlit as st
import pickle
import os

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from src.model import train_model, evaluate_model
from src.load_data import load_data
from src.custom_buttons import button_dropdown_list, numberNone_input, get_model_params


def page5():
    # Set title
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
        "Extra Trees": ExtraTreesRegressor,
        "Random Forest": RandomForestRegressor,
    }
    class_model_name = st.selectbox("Type of model", class_models.keys())
    class_model = class_models[class_model_name]
    model_params = get_model_params(class_model)

    # Train model
    if st.button("Train your model"):
        with st.spinner("Please wait..."):
            model_dict = train_model(class_model, X_train, 
                                     y_train, model_params=model_params)
        # Download model
        st.download_button("Download your model", pickle.dumps(model_dict["model"]),
                           file_name="model.pkl")
        
    button_dropdown_list("Cross validation")
    if st.session_state["Cross validation"]:
        n_splits = st.number_input("Number of splits", min_value=2)
        random_state = numberNone_input("Random state", min_value=0)
        if st.button('Compute cross validation score'):
            with st.spinner("Please wait..."):
                score = evaluate_model(
                    class_model, 
                    X_train, y_train, model_params=model_params,
                    n_splits=n_splits, random_state=random_state)
            st.markdown(f"The average coefficient of determination is R<sup>2</sup>={round(score, 3)}", 
                        unsafe_allow_html=True)