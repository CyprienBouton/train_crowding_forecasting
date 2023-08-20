import streamlit as st
from src.visualize import (price_weekday, price_date, price_hour)

def page3():
    st.markdown("<h1>Data Visualization</h1>", unsafe_allow_html=True)
    parameter_to_visualized = {
        "Date": price_date,
        "Day": price_weekday,
        "Hour": price_hour,
    }

    selected_visualization = st.selectbox("Choose a parameter:", parameter_to_visualized.keys())
    st.pyplot(parameter_to_visualized[selected_visualization]())