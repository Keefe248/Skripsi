from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "model_terbaik.pkl")

# input user
input_data = pd.DataFrame([{
    "age": 25,
    "sex": "male",
    "bmi": 28.5,
    "children": 1,
    "smoker": "no",
    "region": "southeast"
}])

# prediksi
prediction = model.predict(input_data)

print(prediction[0])