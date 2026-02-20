import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

print("Starting training...")

# Load dataset
df = pd.read_csv("dataset.csv")

# Drop unnecessary columns
df = df.drop(["Unnamed: 0", "track_id", "track_name", "album_name", "artists", "track_genre"], axis=1)


# Convert explicit to int
df["explicit"] = df["explicit"].astype(int)

# Define X and y
X = df.drop("popularity", axis=1)
y = df["popularity"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
pickle.dump(model, open("spotify_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully!")
