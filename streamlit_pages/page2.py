import streamlit as st
import pandas as pd
import pickle

def page2():
    """
    Page showing information about dataset.
    """
    st.markdown("<h1>Dataset</h1>",
     unsafe_allow_html=True)
    st.markdown("""
    The data comes from infra-red sensors located above each door of the 
        rolling stocks (NAT, R2N) in Île-de-France, measuring the number of 
        alighting and boarding passengers per door. This data is captured in 
        real time and is accessible only at the train scale for this challenge.
            
    The columns, i.e., the features, are split into 6 contextual variables such 
        as day, train id, etc. and 6 lag variables:
                 
    Context Variables
    - date: date of train passage
    - train: id of the train (unique by day)
    - station: station id
    - hour: time slot
    - way: wether the train is going toward Paris (way is 0) or suburb (way is 1)
    - composition: number of train unit
                
    Lags variables
    - p1q0: occupancy rate of the previous train at the same station
    - p2q0: occupancy rate of the second previous train at the same station
    - p3q0: occupancy rate of the third train at the same station
    - p0q1: occupancy rate of the same train k at the previous station
    - p0q2: occupancy rate of the same train k at the second previous station
    - p0q3: occupancy rate of the same train k at the third previous station
    """)