from sklearn.linear_model import LinearRegression
from src.evaluate_model import evaluate_model
from src.load_data import load_data
from src.preprocess import preprocessing

X, y, _ = load_data()
preprocessing(X)

import streamlit as st
from streamlit_pages import main_page, page2, page3, page4

if __name__=="__main__":
    page_names_to_funcs = {
    "General Description": main_page.main_page,
    "Dataset": page2.page2,
    "Data visualization": page3.page3,
    "Occupancy rate forecasting": page4.page4,
    }

    selected_page = st.sidebar.selectbox("Go to page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()