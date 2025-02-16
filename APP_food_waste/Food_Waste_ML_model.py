import pandas as pd 
import numpy as np
import streamlit as st
import pickle

with open("APP_food_waste/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
# Load the trained model
GBR = pickle.load(open("APP_food_waste/Gradient_boosting_reg.pkl", "rb"))

st.title("Food Waste Prediction App")
st.write("## Enter details to predict the amount of food waste")


scale_cols=["meals_served","kitchen_staff","temperature_C","humidity_percent","past_waste_kg","special_event"]
cat_cols=["staff_experience","waste_category"]
col1, col2 = st.columns(2)
with col1:

# User inputs
    meals_served = st.number_input("No of meals served", min_value=1, max_value=4500, value=300)
    kitchen_staff = st.number_input("No of working staff ", min_value=4, max_value=22, value=12)
    temperature_C = st.number_input("Temperature C", min_value=0, max_value=55, value=23)
    humidity_percent = st.number_input("Humidity", min_value=30, max_value=90, value=60)
    past_waste_kg = st.number_input("past waste (KG)", min_value=0, max_value=60, value=27)
    special_event = st.selectbox("Special Event", ["No", "Yes"])  
    staff_experience = st.selectbox("Staff Experience", ["Beginner", "Intermediate","EXPERT"])
    waste_category = st.selectbox("waste category", ["meat", "dairy","vegetables","grains"])


# **Correctly Encode Categorical Features**
experience_map = {"Beginner": 0, "Intermediate": 1, "EXPERT": 2}
waste_category_map = {"meat": 0, "dairy": 1, "vegetables": 2, "grains": 3}
special_event_map={"Yes":1,"No":0}
staff_experience_encoded = experience_map[staff_experience]
waste_category_encoded = waste_category_map[waste_category]
special_event_encoded = special_event_map[special_event]

with col2:
        # Prediction
    if st.button("Predict"):        

    # Create DataFrame
        user_data = np.array([[meals_served, kitchen_staff, temperature_C, humidity_percent, past_waste_kg, 
                        special_event_encoded, staff_experience_encoded, waste_category_encoded]])    
        user_df = pd.DataFrame(user_data, columns=scale_cols + cat_cols)
        user_df[scale_cols] = user_df[scale_cols].astype(float)  # Keep `special_event` numeric
        user_df[cat_cols] = user_df[cat_cols].astype(int)  # Encode categorical variables only


        prediction = GBR.predict(user_df)
        st.write("### Prediction:", "Food Waste amount (kg) is :",prediction )
