import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor

# load model JSON
model = XGBRegressor()
model.load_model("model/model_terbaik.json")

# load fitur input
fitur_input = joblib.load("model/fitur_input.pkl")

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

    # dataframe input
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    # one hot encoding
    input_data = pd.get_dummies(input_data)

    # samakan kolom dengan training
    input_data = input_data.reindex(
        columns=fitur_input,
        fill_value=0
    )

    # prediksi
    prediction = model.predict(input_data)

    st.success(f"Prediksi biaya medis: ${prediction[0]:,.2f}")