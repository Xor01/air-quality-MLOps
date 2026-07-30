import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Riyadh Air Quality Predictor",
    page_icon="🌍",
)

st.title("🌍 Riyadh Air Quality Predictor")
st.write("Predict whether the next hour will have high PM2.5 pollution.")

st.header("Input Features")

pm2_5 = st.number_input("PM2.5", min_value=0.0, value=25.0)
pm10 = st.number_input("PM10", min_value=0.0, value=40.0)
temperature_2m = st.number_input("Temperature (°C)", value=35.0)
relative_humidity_2m = st.slider("Relative Humidity (%)", 0, 100, 30)
wind_speed_10m = st.number_input("Wind Speed (m/s)", min_value=0.0, value=10.0)

hour = st.slider("Hour", 0, 23, 12)
day_of_week = st.slider("Day of Week", 0, 6, 2)

pm2_5_lag_1 = st.number_input("PM2.5 Lag 1", min_value=0.0, value=24.0)
pm2_5_lag_3 = st.number_input("PM2.5 Lag 3", min_value=0.0, value=23.0)
pm2_5_rolling_mean_6 = st.number_input(
    "PM2.5 Rolling Mean (6h)",
    min_value=0.0,
    value=24.0,
)

if st.button("Predict"):

    payload = {
        "pm2_5": pm2_5,
        "pm10": pm10,
        "temperature_2m": temperature_2m,
        "relative_humidity_2m": relative_humidity_2m,
        "wind_speed_10m": wind_speed_10m,
        "hour": hour,
        "day_of_week": day_of_week,
        "pm2_5_lag_1": pm2_5_lag_1,
        "pm2_5_lag_3": pm2_5_lag_3,
        "pm2_5_rolling_mean_6": pm2_5_rolling_mean_6,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        st.success("Prediction completed!")

        st.metric(
            "Prediction",
            "High Pollution"
            if result["prediction"] == 1
            else "Normal",
        )

        st.metric(
            "Probability",
            f"{result['prediction_probability']:.4f}",
        )

        st.write("Model:", result["model"])
        st.write("Threshold:", result["threshold"])

    except requests.exceptions.RequestException as error:
        st.error(f"API Error: {error}")