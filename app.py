import pandas as pd
import streamlit as st
import numpy as np
import pickle
import warnings
import tempfile
import os
warnings.filterwarnings("ignore")


from absenteeism_module import *


def Input_Output():
    data = st.file_uploader("Please Upload Your Dataset Here", type={"csv", "txt"})
    df = None
    model = None
    
    if data is not None:
        df = pd.read_csv(data)
        st.write(df)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                df.to_csv(tmp.name, index=False)
                model = absenteeism_model('model', 'scaler')
                model.load_and_clean_data(tmp.name)
                os.unlink(tmp.name)  # Delete the temp file

    result = ""

    if st.button("Click here to view sample file"):
        smp = pd.read_csv('Absenteeism_new_data.csv')
        st.write(smp)
        
    if st.button("Click here to Predict"):
        result = model.predicted_outputs()
        st.balloons()     
    st.success('The output is as follows: ')
    st.write(result)


if __name__ ==  '__main__':
    Input_Output()
