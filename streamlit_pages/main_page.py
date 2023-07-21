import streamlit as st
from . import page2, page3, page4

def main_page():
    """
    Show the different section of the apps.
    """
    st.markdown("<h1>SNCF-Transilien Data Challenge</h1>", unsafe_allow_html=True)
    st.markdown("""
    The aim of this challenge is to give SNCF-Transilien the tools to \
        provide an accurate train occupancy rate forecasting. Thus, \
        deliver precise real time crowding information (RTCI) to its \
        passengers through digital services..<br/><br/>
    There are 3 sections in this app:
    - Dataset
    - Data visualization
    - Occupancy rate forecasting
    """, unsafe_allow_html=True)