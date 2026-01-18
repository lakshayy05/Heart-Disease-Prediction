import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="Heart Stroke Prediction", layout="centered")

# --- LOAD SAVED ASSETS ---
try:
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
except FileNotFoundError:
    st.error("Error: One or more model files (.pkl) are missing! Please ensure 'KNN_heart.pkl', 'scaler.pkl', and 'columns.pkl' are in this folder.")
    st.stop()

st.title("❤️ Heart Stroke Risk Predictor")
st.write("Enter your medical details below to assess heart stroke risk.")

# --- USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 45)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0, 0.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# --- PREDICTION LOGIC ---
if st.button("Predict Risk"):
    try:
        # 1. Prepare Raw Input Dictionary
        # Note: We structure this to match how get_dummies works
        input_data = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex': sex,
            'ChestPainType': chest_pain,
            'RestingECG': resting_ecg,
            'ExerciseAngina': exercise_angina,
            'ST_Slope': st_slope
        }
        
        # 2. Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 3. One-Hot Encoding (Convert categorical text to numbers)
        # This creates columns like 'Sex_M', 'Sex_F', etc.
        input_df_encoded = pd.get_dummies(input_df)
        
        # 4. Align Columns with Training Data
        # This is CRITICAL. It adds missing columns (fill=0) and removes extra ones
        # to ensure the model sees exactly what it expects.
        input_df_encoded = input_df_encoded.reindex(columns=expected_columns, fill_value=0)
        
        # 5. Scale the Data
        # We use .values to send a numpy array, which avoids the "Feature Name" warning
        input_scaled = scaler.transform(input_df_encoded.values)
        
        # 6. Predict
        prediction = model.predict(input_scaled)
        
        # 7. Display Result
        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Stroke detected. Please consult a doctor.")
        else:
            st.success("✅ Low Risk. Your heart health appears normal.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- SIDEBAR ---
st.sidebar.info("This app uses a K-Nearest Neighbors (KNN) model to predict heart failure risk.")