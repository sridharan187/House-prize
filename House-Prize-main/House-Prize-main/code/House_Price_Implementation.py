# Import necessary libraries
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Web app title
st.title("House Price Prediction App")

# Step 1: Load the dataset
housing_data = fetch_california_housing(as_frame=True)  # Fetch dataset as a pandas DataFrame
df = housing_data.frame

features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']
target = 'MedHouseVal'

X = df[features]
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 4: User Input
st.header("Enter House Details for Price Prediction")

# Input fields
med_inc = st.number_input("Median Income (in $10,000s)", min_value=0.0, max_value=20.0, step=0.1)
house_age = st.slider("House Age (in years)", min_value=0, max_value=100, step=1)
ave_rooms = st.number_input("Average Number of Rooms", min_value=1.0, max_value=20.0, step=0.1)
ave_bedrooms = st.number_input("Average Number of Bedrooms", min_value=1.0, max_value=10.0, step=0.1)
population = st.number_input("Population in the Block", min_value=1, max_value=5000, step=1)

# Predict button
if st.button("Predict"):
    # Normalize the input values
    input_data = scaler.transform([[med_inc, house_age, ave_rooms, ave_bedrooms, population]])

    # Make prediction
    prediction = model.predict(input_data)[0]
    st.subheader(f"Predicted Median House Price: ${prediction * 100000:.2f}")



