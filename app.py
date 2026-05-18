import joblib
import pandas as pd

# load model
model = joblib.load("best_model_xgboost.pkl")

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