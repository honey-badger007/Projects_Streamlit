import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle

st.set_page_config(page_title="Food Waste Prediction", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.page_link("Food_waste/pages/prediction.py", label="🔮 Food Waste Prediction")
st.sidebar.page_link("Food_waste/pages/visualization.py", label="📊 Data Visualization")

st.title("Welcome to the Food Waste Analysis & Prediction App!")
st.write(
    """
    📌 Use the **sidebar** to navigate:
    - 🔮 **Food Waste Prediction**: Predict food waste based on input data.
    - 📊 **Data Visualization**: Explore and analyze the dataset visually.
    """
)

