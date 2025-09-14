import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
try:
    model = joblib.load('Earthquake_forecasting.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'Earthquake_forecasting.pkl' is in the same directory as the app or provide the full path.")
    st.stop()

st.title('Earthquake Category Prediction')

st.write("Enter the features to predict the earthquake category.")

# Add input fields for the features used in the model
# Based on your feature selection, these are 'mag', 'magNst', and 'nst'
mag = st.number_input('Magnitude (mag)', min_value=0.0, max_value=10.0, value=4.5)
magNst = st.number_input('Magnitude Station Count (magNst)', min_value=0.0, value=10.0)
nst = st.number_input('Station Count (nst)', min_value=0.0, value=20.0)


# Create a button to trigger prediction
if st.button('Predict Category'):
    # Create a DataFrame with the input features
    # The column names must match the features used during training
    input_data = pd.DataFrame([[mag, magNst, nst]], columns=['mag', 'magNst', 'nst'])

    # Make prediction using the processed input data
    prediction = model.predict(input_data)


    # Map the predicted category back to the original labels
    # Reconstructing mapping based on observations (0: Low, 1: Major, 2: Moderate, 3: Strong)
    category_mapping = {0: 'Low', 1: 'Major', 2: 'Moderate', 3: 'Strong'} # Reconstructed mapping based on observations


    predicted_category = category_mapping.get(prediction[0], 'Unknown')


    st.success(f'Predicted Earthquake Category: {predicted_category}')
