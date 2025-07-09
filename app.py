import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Predictive Maintenance - Flavi Dairy", layout="centered")
st.title("🛠️ Predictive Maintenance using Sensor Data")

st.markdown("This app uses vibration and temperature data to predict equipment condition.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("sensor_data.csv")

df = load_data()

# Train model
X = df[["vibration", "temperature"]]
y = df["wear_tear"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sidebar input
st.sidebar.header("📥 Input Sensor Values")
vibration = st.sidebar.slider("Vibration Level", 2.0, 12.0, 5.0, step=0.1)
temperature = st.sidebar.slider("Temperature (°C)", 30.0, 90.0, 45.0, step=0.5)

if st.sidebar.button("Predict Condition"):
    input_data = pd.DataFrame([[vibration, temperature]], columns=["vibration", "temperature"])
    prediction = model.predict(input_data)[0]
    condition = {0: "✅ Normal", 1: "❌ Fault"}
    st.subheader("🔍 Prediction Result")
    st.success(f"Equipment Status: **{condition[prediction]}**")
    st.write("### 📊 Input Values")
    st.write(input_data)

# Batch prediction
st.write("### 📤 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV with 'vibration' and 'temperature'", type=["csv"])
if uploaded_file:
    df_up = pd.read_csv(uploaded_file)
    if "vibration" in df_up.columns and "temperature" in df_up.columns:
        preds = model.predict(df_up[["vibration", "temperature"]])
        df_up["Predicted Status"] = [ {0: "Normal", 1: "Fault"}[i] for i in preds ]
        st.dataframe(df_up)
    else:
        st.error("CSV must have 'vibration' and 'temperature' columns.")
