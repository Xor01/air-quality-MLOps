import os 
 
import requests 
import streamlit as st 
 
API_URL = os.getenv("API_URL", "http://localhost:8000") 
st.set_page_config(page_title="Riyadh Air Quality", layout="centered") 
st.title("Riyadh Air Quality Risk") 
st.write("Predict whether the next hour will have high PM2.5 pollution.")
with st.form("prediction-form"): 
    pm2_5 = st.number_input("Current PM2.5", 0.0, 1000.0, 25.0) 
    pm10 = st.number_input("Current PM10", 0.0, 1500.0, 60.0) 
    temperature = st.number_input("Temperature", -20.0, 65.0, 32.0) 
    humidity = st.number_input("Relative humidity", 0.0, 100.0, 25.0) 
    wind = st.number_input("Wind speed", 0.0, 200.0, 12.0) 
    hour = st.slider("Hour", 0, 23, 12) 
    day = st.slider("Day of week", 0, 6, 2) 
    lag_1 = st.number_input("PM2.5 one hour ago", 0.0, 1000.0, 24.0) 
    lag_3 = st.number_input("PM2.5 three hours ago", 0.0, 1000.0, 22.0) 
    rolling = st.number_input("Six-hour PM2.5 average", 0.0, 1000.0, 23.0) 
    submitted = st.form_submit_button("Predict") 
 
if submitted: 
    payload = { 
        "pm2_5": pm2_5, "pm10": pm10, "temperature_2m": temperature, 
        "relative_humidity_2m": humidity, "wind_speed_10m": wind, 
        "hour": hour, "day_of_week": day, "pm2_5_lag_1": lag_1, 
        "pm2_5_lag_3": lag_3, "pm2_5_rolling_mean_6": rolling, 
    } 
    try: 
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10) 
        response.raise_for_status() 
        result = response.json() 
        st.metric("High-risk probability", f"{result['prediction_probability']:.1%}") 

        st.badge(f"Risk level: {result['risk_level']}", color="green" if result['risk_level'] == "normal" else "red", width='stretch')
        st.metric(
            "Probability",
            f"{result['prediction_probability']:.4f}",
        )

        st.write("Model:", result["model"])
        st.write("Threshold:", result["threshold"])
    except requests.RequestException as error: 
        st.error(f"API request failed: {error}") 
 
st.divider() 
try: 
    status = requests.get(f"{API_URL}/health", timeout=5).json() 
    st.caption(f"API status: {status['status']}") 
except requests.RequestException: 
    st.caption("API status: unavailable") 