import streamlit as st
import select_randomsamples
import pandas as pd
import data_preprocessing
from datetime import datetime
import predict_mydata
st.title("Ensemble Learning for IDS")
st.write("-An ML application that analyzes the network traffic and detects the intrusions in the network")
threshold = st.slider("Select the Threshold", min_value=1, max_value=14, value=1)
st.write("Please click on Predict the Traffic to obtain the predictions")
if st.button("Predict the Traffic"):
    data = select_randomsamples.get_data()
    result_data = data.copy()
    processed_data = data_preprocessing.dp_preprocessing(data)
    predict_data = processed_data.drop(columns=["class"], axis=1)
    predicted_data = predict_mydata.predict_mydata(predict_data, threshold)
    result_data = pd.concat([result_data, predicted_data], axis=1)
    result_data["attack"] = result_data["attack"].apply(lambda x:"normal" if x=="normal" else "attack")
    st.write(result_data)