import streamlit as st
from . import page2, page3, page4

def main_page():
    """
    Show the different section of the apps.
    """
    st.markdown("<h1>SNCF-Transilien Data Challenge</h1>", unsafe_allow_html=True)
    st.markdown("""
    </p><h2>Context</h2>
        This project is part of a Data Challenge provided by ENS. Each year, the school 
        organizes machine learning challenges from data provided by public services, companies or 
        laboratories. These challenges are free and open to anyone.
        You can find this data challenge
            <a href="https://challengedata.ens.fr/challenges/89">here</a>.
    </p>
    
    <p><h2>Goal</h2>
        The aim of this challenge is to give SNCF-Transilien 
        (Railway network operator covering the Paris region) the tools to
        provide an accurate train occupancy rate forecasting. This tool will allow
        the travelers to know how busy the train will be when they board it. This challenge
        focus solely on forecasting the occupacy rates at the next station.
    </p>
    
    <p><h2>Summary</h2>
        <ul>
            <li>Dataset</li>
            <li>Visualization</li>
            <li>Occupancy rate forecasting</li>
        </ul>
    </p>
    """, unsafe_allow_html=True)