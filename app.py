import streamlit as st
import xgboost as xgb
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define target columns
target_columns = ['bearings', 'wpump', 'radiator', 'exvalve', 'acmotor']

# Load XGBoost models
xgb_models = {}
for target in target_columns:
    model = xgb.XGBClassifier()
    model.load_model(f'xgboost_model_{target}.json')
    xgb_models[target] = model

# Load Label Encoders
label_encoders = {}
for target in target_columns:
    with open(f'label_encoder_{target}.json', 'r') as f:
        data = json.load(f)
        le = LabelEncoder()
        le.classes_ = np.array(data['classes'])
        label_encoders[target] = le

# Define feature columns
feature_columns = ['outlet_temp', 'motor_power', 'torque', 'rpm', 'gaccx', 'gaccz', 'gaccy', 'haccy']

# Function to make predictions
def predict(features):
    input_df = pd.DataFrame([features], columns=feature_columns)
    predictions = {}
    for target in target_columns:
        model = xgb_models[target]
        le = label_encoders[target]
        prediction_encoded = model.predict(input_df)
        prediction = le.inverse_transform(prediction_encoded)
        predictions[target] = prediction[0]
    return predictions



# Streamlit UI
st.title('Predictive Maintenance Of AC')
image_path = "AC.jpg"
st.image(image_path, width=500)

st.write("Enter the features to get predictions:")

# Input fields with limits
outlet_temp = st.number_input('Outlet Temperature', min_value=76.9, max_value=173.0, format="%.1f")
motor_power = st.number_input('Motor Power', min_value=1400, max_value=19500, format="%d")
torque = st.number_input('Torque', min_value=13.2, max_value=93.5, format="%.1f")
rpm = st.number_input('RPM', min_value=480, max_value=2520, format="%d")
gaccx = st.number_input('GACC X', min_value=0.54, max_value=0.73, format="%.2f")
gaccy = st.number_input('GACC Y', min_value=0.27, max_value=0.46, format="%.2f")
gaccz = st.number_input('GACC Z', min_value=1.73, max_value=9.21, format="%.2f")
haccy = st.number_input('HACC Y', min_value=1.27, max_value=1.46, format="%.2f")

# Button to make predictions
if st.button('Predict'):
    features = [outlet_temp, motor_power, torque, rpm, gaccx, gaccz, gaccy, haccy]
    predictions = predict(features)
    for target, prediction in predictions.items():
        st.write(f"Prediction for {target}: {prediction}")
