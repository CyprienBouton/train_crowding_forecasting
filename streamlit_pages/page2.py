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
    The data comes from infra-red sensors located above each door of the \
        rolling stocks (NAT, R2N) in Île-de-France, measuring the number of \
        alighting and boarding passengers per door. This data is captured in \
        real time and is accessible only at the train scale for this challenge.
            
    The columns, i.e., the features, are split into 6 contextual variables such \
        as day, train id, etc. and 6 lag variables:
                 
    Context Variables
    - date
    - train
    - station
    - hour
    - way
    - composition 
                
    Lags variables
    - p1q0
    - p2q0
    - p3q0
    - p0q1
    - p0q2
    - p0q3
                
    We obtain the following dataset:
    """)
    import os.path as path
    data = pd.read_csv('datasets/Xtrain_hgcGIrA.csv')
    st.write(data)