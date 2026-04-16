# ==========================================
# ✈️ AirFare Vista - Final Streamlit App
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import date

# ==========================================
# 🔧 PATH HANDLING (COLAB + LOCAL + CLOUD)
# ==========================================
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = "/content/drive/MyDrive/AirFair-Vista"

BACKEND_PATH = os.path.join(BASE_DIR, "backend")

if BACKEND_PATH not in sys.path:
    sys.path.append(BACKEND_PATH)

# ==========================================
# 🔗 IMPORT MODEL FUNCTION
# ==========================================
from preprocessor import predict_flight_price

# ==========================================
# 🎨 PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AirFare Vista",
    page_icon="✈️",
    layout="wide"
)

# ==========================================
# 🎨 FINAL THEME-SAFE CSS (DARK + LIGHT)
# ==========================================
st.markdown("""
<style>

/* ===== LABEL FIX ===== */
label {
    color: inherit !important;
}

/* ===== INPUT BOX ===== */
input {
    color: inherit !important;
}

/* ===== LIGHT MODE ===== */
@media (prefers-color-scheme: light) {

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: #0f172a !important;
    }

    div[data-baseweb="menu"] {
        background-color: #ffffff !important;
    }

    li[role="option"] {
        color: #0f172a !important;
    }
}

/* ===== DARK MODE ===== */
@media (prefers-color-scheme: dark) {

    div[data-baseweb="select"] > div {
        background-color: #262730 !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: #ffffff !important;
    }

    div[data-baseweb="menu"] {
        background-color: #1e1e2e !important;
    }

    li[role="option"] {
        color: #ffffff !important;
    }

    li[role="option"]:hover {
        background-color: #2d3a5e !important;
    }
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 🏷️ HEADER
# ==========================================
st.title("✈️ Flight Price Predictor")
st.markdown("### ML-powered Airline Fare Prediction System")

# ==========================================
# ✈️ SECTION A — ROUTE
# ==========================================
st.subheader("✈️ Route Details")

col1, col2 = st.columns(2)

source = col1.selectbox(
    "Source City",
    ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"]
)

destination = col2.selectbox(
    "Destination City",
    ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"]
)

# ==========================================
# 🛫 SECTION B — FLIGHT DETAILS
# ==========================================
st.subheader("🛫 Flight Details")

col3, col4 = st.columns(2)

airline = col3.selectbox(
    "Airline",
    ["IndiGo", "Air India", "SpiceJet", "Vistara", "GoAir"]
)

stops = col4.selectbox(
    "Total Stops",
    [0, 1, 2]
)

# ==========================================
# 📅 SECTION C — DATE (FUTURE ONLY)
# ==========================================
today = date.today()

travel_date = st.date_input(
    "📆 Travel Date",
    value=today,
    min_value=today
)

journey_day = travel_date.day
journey_month = travel_date.month
journey_weekday = travel_date.weekday()

# ==========================================
# ⏱️ EXTRA INPUTS
# ==========================================
col5, col6 = st.columns(2)

dep_hour = col5.slider("Departure Hour", 0, 23, 10)
passengers = col6.number_input("Passengers", 1, 9, 1)

# ==========================================
# ⚠️ INFO
# ==========================================
st.info("ℹ️ Model trained on historical data. Predictions are approximate.")

# ==========================================
# 🔮 PREDICTION
# ==========================================
if st.button("Predict Price"):

    if source == destination:
        st.error("Source and Destination cannot be the same.")
    else:
        try:
            user_input = {
                "journey_day": journey_day,
                "journey_month": journey_month,
                "journey_weekday": journey_weekday,
                "dep_hour": dep_hour,
                "Source": source,
                "Destination": destination,
                "Airline": airline,
                "Total_Stops": stops
            }

            price = predict_flight_price(user_input)

            st.success(f"💰 Estimated Ticket Price: ₹ {round(price, 2)}")

        except Exception as e:
            st.error(f"Error: {e}")
