import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle

option = st.sidebar.selectbox("Choose a page", ("Food Waste Prediction", "Data Visualization"))

if option == "Food Waste Prediction":
    # Add your code to load the prediction page content here
    st.title("Food Waste Prediction")
    # Load prediction content

elif option == "Data Visualization":
    # Add your code to load the visualization page content here
    st.title("Data Visualization")
    # Load visualization content
