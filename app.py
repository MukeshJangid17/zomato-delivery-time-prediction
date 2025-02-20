import streamlit as st
import pandas as pd
import mlflow.pyfunc
import json
import joblib
from sklearn.pipeline import Pipeline
from mlflow import MlflowClient

# Initialize Dagshub
import dagshub
import mlflow.client

dagshub.init(repo_owner='mukeshjangid7877', repo_name='zomato-delivery-time-prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/mukeshjangid7877/zomato-delivery-time-prediction.mlflow")

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

def load_transformer(transformer_path):
    return joblib.load(transformer_path)

# Load model info
model_name = load_model_information("run_information.json")['model_name']
stage = "Staging"
model_path = f"models:/{model_name}/{stage}"
model = mlflow.sklearn.load_model(model_path)

# Load preprocessor
preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# Build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', model)
])

# Streamlit App
st.title("Zomato Delivery Time Prediction")

# Input fields
age = st.selectbox("Age of Delivery Partner", [18, 20, 25, 30, 35, 40, 45])
ratings = st.selectbox("Delivery Partner Ratings", [1, 2, 3, 4, 5])
weather = st.selectbox("Weather Conditions", ["Sunny", "Cloudy", "Rainy", "Stormy"])
traffic = st.selectbox("Traffic Conditions", ["Low", "Medium", "High"])
vehicle_condition = st.selectbox("Vehicle Condition", [0, 1, 2])
type_of_order = st.selectbox("Type of Order", ["Snack", "Meal", "Beverage", "Dessert"])
type_of_vehicle = st.selectbox("Type of Vehicle", ["Motorcycle", "Scooter", "EV"])
multiple_deliveries = st.selectbox("Multiple Deliveries", [0, 1, 2, 3])
festival = st.selectbox("Festival Season", ["Yes", "No"])
city_type = st.selectbox("City Type", ["Metro", "Urban", "Rural"])
is_weekend = st.selectbox("Is Weekend?", ["Yes", "No"])
pickup_time_minutes = st.slider("Pickup Time (minutes)", 1, 60, 10)
order_time_of_day = st.selectbox("Order Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
distance = st.number_input("Distance (km)", min_value=0.5, max_value=50.0, value=5.0, step=0.2)
distance_type = st.selectbox("Distance Type", ["Short", "Medium", "Long"])

# Predict Button
if st.button("Predict Delivery Time"):
    input_data = pd.DataFrame({
        "age": [age],
        "ratings": [ratings],
        "weather": [weather],
        "traffic": [traffic],
        "vehicle_condition": [vehicle_condition],
        "type_of_order": [type_of_order],
        "type_of_vehicle": [type_of_vehicle],
        "multiple_deliveries": [multiple_deliveries],
        "festival": [festival],
        "city_type": [city_type],
        "is_weekend": [is_weekend],
        "pickup_time_minutes": [pickup_time_minutes],
        "order_time_of_day": [order_time_of_day],
        "distance": [distance],
        "distance_type": [distance_type],
    })
    
    # Apply preprocessing
    input_transformed = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_transformed)
    
    st.success(f"Predicted Delivery Time: {round(prediction[0], 2)} minutes")
