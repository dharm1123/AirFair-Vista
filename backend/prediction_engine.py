import joblib
import numpy as np
import sys

sys.path.append("/content/drive/MyDrive/AirFair-Vista/backend")

from preprocessing import engineer_features, preprocess_input


BASE_PATH = "/content/drive/MyDrive/AirFair-Vista"

pipeline = joblib.load(
    f"{BASE_PATH}/models/flight_price_prediction_pipeline.pkl"
)

model = pipeline["model"]
features = pipeline["features"]


def predict_flight_price(user_input):

    engineered = engineer_features(user_input)

    processed = preprocess_input(engineered, features)

    prediction_log = model.predict(processed)

    price = np.expm1(prediction_log)

    return float(price[0])
