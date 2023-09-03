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
        alighting and boarding passengers per door.
            
    The columns, i.e., the features, are split into 6 contextual variables such 
        as day, train id, etc. and 6 lag variables:
                 
    <h4>Context Variables</h4>
    - date: date of train passage
    - train: id of the train (unique by day)
    - station: station id
    - hour: time slot
    - way: wether the train is going toward Paris (way is 0) or suburb (way is 1)
    - composition: number of train unit
                
    <h4>Lags variables</h4>
    - p1q0: Occupancy rate of the previous train k-1 at the same station s
    - p2q0: Occupancy rate of the second previous k-2 train at the same station s
    - p3q0: Occupancy rate of the third train k-3 at the same station s
    - p0q1: Occupancy rate of the same train k at the previous station s-1
    - p0q2: Occupancy rate of the same train k at the second previous station s-2
    - p0q3: Occupancy rate of the same train k at the third previous station s-3
    """)