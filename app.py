import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model_streamlit.pkl")


st.set_page_config(page_title="Predictive Maintenance - Flavi Dairy", layout="centered")
st.title("🛠️ Predictive Maintenance using Sensor Data")
st.markdown("This app uses vibration and temperature data to predict equipment condition (Normal or Fault).")

st.sidebar.header("📥 Input Sensor Values")
vibration = st.sidebar.slider("Vibration Level (Hz)", 2.0, 12.0, 5.0, step=0.1)
temperature = st.sidebar.slider("Temperature (°C)", 30.0, 90.0, 45.0, step=0.5)

if st.sidebar.button("Predict Condition"):
    input_data = pd.DataFrame([[vibration, temperature]], columns=["vibration", "temperature"])
    prediction = model.predict(input_data)[0]
    condition = {0: "✅ Normal", 1: "❌ Fault"}
    st.subheader("🔍 Prediction Result")
    st.success(f"Equipment Status: **{condition[prediction]}**")
    st.write("### 📊 Input Values")
    st.write(input_data)

st.write("### 📤 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with 'vibration' and 'temperature' columns", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "vibration" in df.columns and "temperature" in df.columns:
        preds = model.predict(df[["vibration", "temperature"]])
        df["Predicted Status"] = [ {0: "Normal", 1: "Fault"}[i] for i in preds ]
        st.dataframe(df)
    else:
        st.error("CSV must have 'vibration' and 'temperature' columns.")
