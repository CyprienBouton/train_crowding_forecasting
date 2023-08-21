import streamlit as st
from src.visualize import (price_weekday, price_date, price_hour, price_composition,
                           price_train_id, price_station)

def page3():
    st.markdown("<h1>Visualization</h1>", unsafe_allow_html=True)
    parameter_to_visualized = {
        "Date": price_date,
        "Day": price_weekday,
        "Hour": price_hour,
        "Composition": price_composition,
        "Train ids": price_train_id,
        "Station": price_station,
    }

    selected_visualization = st.selectbox("Choose a parameter:", parameter_to_visualized.keys())
    st.pyplot(parameter_to_visualized[selected_visualization]())