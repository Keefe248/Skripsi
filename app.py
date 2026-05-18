import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/model_terbaik.pkl")

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