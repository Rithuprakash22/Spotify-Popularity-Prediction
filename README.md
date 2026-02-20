🎵 Spotify Popularity Prediction

A Machine Learning web application that predicts the popularity score of a song based on its audio features.
The model is trained using Scikit-learn and deployed using Streamlit.


📌 Project Overview

This project builds an end-to-end ML pipeline to predict Spotify song popularity using audio characteristics such as danceability, energy, loudness, tempo, and more.

The system includes:
                    Data preprocessing
                    Feature scaling
                    Model training
                    Model evaluation
                    Web deployment

📊 Dataset

The dataset contains Spotify track features including:
duration_ms
explicit
danceability
energy
key
loudness
mode
speechiness
acousticness
instrumentalness
liveness
valence
tempo
time_signature
popularity (Target)

Unnecessary columns were removed during preprocessing.


🛠 Machine Learning Pipeline:

Data Cleaning & Feature Selection
Train-Test Split (80% / 20%)
Feature Scaling using StandardScaler
Baseline Model: Linear Regression
Final Model: Random Forest Regressor
Evaluation using MAE, RMSE, and R²
Model saved using Pickle

🌐 Streamlit Web Application

The trained model is deployed using Streamlit.

Features:
Sidebar input for song audio features
Real-time prediction
Displays predicted popularity score
Simple and interactive interface
Run the application:
                      streamlit run spotify_app.py

Open in browser


⚙️ Technologies Used:

        Python
        Pandas
        NumPy
        Scikit-learn
        Streamlit
