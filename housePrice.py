# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# 2. Train model (simple Linear Regression)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Streamlit UI
st.title("🏡 House Price Prediction App")

st.write("Enter house details below to predict the price:")

# User inputs
MedInc = st.number_input("Median Income (in 10,000s)", min_value=0.0, max_value=20.0, value=5.0)
HouseAge = st.slider("House Age (years)", 1, 50, 20)
AveRooms = st.number_input("Average Rooms", min_value=1.0, max_value=15.0, value=5.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=1.0, max_value=10.0, value=2.0)
Population = st.number_input("Population in area", min_value=100, max_value=50000, value=1000)
AveOccup = st.number_input("Average Occupancy", min_value=1.0, max_value=10.0, value=3.0)
Latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
Longitude = st.slider("Longitude", -124.0, -114.0, -120.0)

# 4. Prepare input for prediction
input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]],
                          columns=X.columns)

# 5. Predict
prediction = model.predict(input_data)[0]

st.subheader("Predicted House Price")
st.write(f"${prediction * 100000:.2f}")