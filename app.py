import streamlit as st
import pandas as pd
import joblib

# load model pipeline lengkap
model = joblib.load("model/model_terbaik.pkl")

st.title("Prediksi Biaya Medis")

# input user
age = st.number_input("Age", 0, 100, 25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 28.5)
children = st.number_input("Children", 0, 10, 1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["southeast", "southwest", "northwest", "northeast"]
)

if st.button("Prediksi"):

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    # pipeline otomatis preprocessing sendiri
    prediction = model.predict(input_data)

    st.success(f"Prediksi biaya medis: ${prediction[0]:,.2f}")