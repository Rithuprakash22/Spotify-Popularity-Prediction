import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("spotify_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Spotify Popularity Predictor", layout="wide")

# ------------------ SIDEBAR INPUTS ------------------
st.sidebar.header("🎛 Enter Song Features")

duration_ms = st.sidebar.number_input("Duration (ms)", value=200000)
explicit = st.sidebar.selectbox("Explicit (0 = No, 1 = Yes)", [0, 1])
danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5)
key = st.sidebar.number_input("Key (0-11)", 0, 11, 5)
loudness = st.sidebar.number_input("Loudness", value=-5.0)
mode = st.sidebar.selectbox("Mode (0 = Minor, 1 = Major)", [0, 1])
speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.1)
acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.3)
instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.0)
liveness = st.sidebar.slider("Liveness", 0.0, 1.0, 0.1)
valence = st.sidebar.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.sidebar.number_input("Tempo", value=120.0)
time_signature = st.sidebar.number_input("Time Signature (1-7)", 1, 7, 4)

predict_button = st.sidebar.button("🎯 Predict Popularity")

# ------------------ MAIN PAGE ------------------
st.markdown("<h1 style='text-align: center;'>🎵 Spotify Popularity Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict song popularity using audio features</h4>", unsafe_allow_html=True)

st.write("")
st.write("")

if predict_button:

    input_data = np.array([[duration_ms, explicit, danceability, energy,
                            key, loudness, mode, speechiness,
                            acousticness, instrumentalness,
                            liveness, valence, tempo, time_signature]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.markdown("<h2 style='text-align: center;'>Prediction Result</h2>", unsafe_allow_html=True)

    st.markdown(
        f"<h1 style='text-align: center; color: green;'>Popularity Score: {int(prediction[0])}</h1>",
        unsafe_allow_html=True
    )

    # Interpretation
    if prediction[0] > 70:
        st.success("🔥 High Popularity Potential!")
    elif prediction[0] > 40:
        st.info("👍 Moderate Popularity Expected")
    else:
        st.warning("⚠ Low Popularity Potential")