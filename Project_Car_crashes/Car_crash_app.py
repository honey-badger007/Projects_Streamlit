import pandas as pd 
import numpy as np
import streamlit as st
import pickle

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the trained model
XGB = pickle.load(open("XGB_model.pkl", "rb"))


# Load the trained model
RF = pickle.load(open("Random_forest.pkl", "rb"))


st.title("Car Crash Severity Prediction")
st.write("## Predict the severity of a car crash based on the given parameters")

# Get the correct feature order from the model
correct_feature_order = XGB.get_booster().feature_names

col1, col2 = st.columns(2)
with col1:
    # User inputs
    Crash_Speed = st.number_input("Crash Speed (km/h)", min_value=10.0, max_value=250.0, value=110.0, format="%.2f")
    Impact_Angle = st.number_input("Impact Angle (degrees)", min_value=0.0, max_value=180.0, value=90.0, format="%.2f")
    Vehicle_Age = st.number_input("Vehicle Age (years)", min_value=0.0, max_value=20.0, value=10.0, format="%.2f")
    Driver_Age = st.number_input("Driver Age", min_value=18.0, max_value=80.0, value=40.0, format="%.2f")
    Driver_Experience = st.number_input("Driver Experience (years)", min_value=0.0, max_value=60.0, value=25.0, format="%.2f")
    Alcohol_Level = st.number_input("Alcohol Level (BAC%)", min_value=0.0, max_value=0.2, value=0.1, format="%.2f")  
    Visibility_Distance = st.number_input("Visibility Distance (m)", min_value=5.0, max_value=500.0, value=250.0, format="%.2f")
    Traffic_Density = st.selectbox("Traffic Density", ['High', 'Low', 'Medium'])
    Time_of_Day = st.selectbox("Time of Day", ['Afternoon', 'Morning', 'Night'])
    Tire_Condition = st.selectbox("Tire Condition", ['Good', 'Worn out'])
    Brake_Condition = st.selectbox("Brake Condition", ['Good', 'Worn out'])
    Vehicle_Type = st.selectbox("Vehicle Type", ['Motorcycle', 'SUV', 'Sedan', 'Truck'])
    Crash_Type = st.selectbox("Crash Type", ['Head-on', 'Rear-end', 'Rollover', 'Side impact'])
    Road_Conditions = st.selectbox("Road Conditions", ['Dry', 'Icy', 'Uneven', 'Wet'])
    Weather_Conditions = st.selectbox("Weather Conditions", ['Clear', 'Fog', 'Rain', 'Snow'])
    Airbag_Deployed = st.selectbox("Airbag Deployed", ['No', 'Yes'])
    Seatbelt_Used = st.selectbox("Seatbelt Used", ['No', 'Yes'])

# Encoding mappings
Traffic_Density_map = {"High": 0, "Low": 1, "Medium": 2}
Time_of_Day_map = {"Afternoon": 0, "Morning": 1, "Night": 2}
Tire_Condition_map = {"Good": 0, "Worn out": 1}
Brake_Condition_map = {"Good": 0, "Worn out": 1}
Vehicle_Type_map = {'Motorcycle': 0, 'SUV': 1, 'Sedan': 2, 'Truck': 3}
Crash_Type_map = {'Head-on': 0, 'Rear-end': 1, 'Rollover': 2, 'Side impact': 3}
Road_Conditions_map = {'Dry': 0, 'Icy': 1, 'Uneven': 2, 'Wet': 3}
Weather_Conditions_map = {'Clear': 0, 'Fog': 1, 'Rain': 2, 'Snow': 3}
Airbag_Deployed_map = {"Yes": 1, "No": 0}
Seatbelt_Used_map = {"Yes": 1, "No": 0}

with col2:
    if st.button("Predict"):
        # Encode categorical variables
        Traffic_Density_encoded = Traffic_Density_map[Traffic_Density]
        Time_of_Day_encoded = Time_of_Day_map[Time_of_Day]
        Tire_Condition_encoded = Tire_Condition_map[Tire_Condition]
        Brake_Condition_encoded = Brake_Condition_map[Brake_Condition]
        Vehicle_Type_encoded = Vehicle_Type_map[Vehicle_Type]
        Crash_Type_encoded = Crash_Type_map[Crash_Type]
        Road_Conditions_encoded = Road_Conditions_map[Road_Conditions]
        Weather_Conditions_encoded = Weather_Conditions_map[Weather_Conditions]
        Airbag_Deployed_encoded = Airbag_Deployed_map[Airbag_Deployed]
        Seatbelt_Used_encoded = Seatbelt_Used_map[Seatbelt_Used]

        # Create dictionary with all features in the correct order
        user_data = {
            'Crash Speed (km/h)': [Crash_Speed],
            'Impact Angle (degrees)': [Impact_Angle],
            'Vehicle Age (years)': [Vehicle_Age],
            'Driver Age': [Driver_Age],
            'Driver Experience (years)': [Driver_Experience],
            'Alcohol Level (BAC%)': [Alcohol_Level],
            'Visibility Distance (m)': [Visibility_Distance],
            'Traffic Density': [Traffic_Density_encoded],
            'Time of Day': [Time_of_Day_encoded],
            'Tire Condition': [Tire_Condition_encoded],
            'Brake Condition': [Brake_Condition_encoded],
            'Vehicle Type': [Vehicle_Type_encoded],
            'Crash Type': [Crash_Type_encoded],
            'Road Conditions': [Road_Conditions_encoded],
            'Weather Conditions': [Weather_Conditions_encoded],
            'Airbag Deployed': [Airbag_Deployed_encoded],
            'Seatbelt Used': [Seatbelt_Used_encoded]
        }

        # Create DataFrame and ensure correct column order
        user_df = pd.DataFrame(user_data)
        user_df = user_df[correct_feature_order]  # Reorder columns to match model
        
        # Make prediction
        prediction = RF.predict(user_df)
        
        # Display prediction
        severity_map = {0:'Fatal', 1:'Minor Injury', 2:'Severe Injury'}
        st.write(f"### Prediction: {severity_map.get(prediction[0], 'Unknown Severity')}")