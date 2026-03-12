import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1 & 2: Load and Clean Dataset ---
# Ensure "DataTest.csv" is in the same folder as this script!
try:
    data = pd.read_csv("DataTest.csv")
    data.drop_duplicates(inplace=True)
    data.fillna(data.mean(numeric_only=True), inplace=True)
    
    # Calculate THI for the training data
    data["THI"] = data["Temperature"] - (0.55 - 0.0055 * data["Humidity"]) * (data["Temperature"] - 14.5)
except FileNotFoundError:
    print("Error: 'DataTest.csv' not found. Please check the file path.")
    exit()

# --- 3: Machine Learning Model Training ---
X = data[["Temperature", "Humidity", "Occupancy"]]
y = data["THI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 4: Prediction & Comfort Logic Function ---
def calculate_thermal_comfort(temp, humidity, occupancy):
    # 1. Prepare data for prediction
    input_df = pd.DataFrame(
        [[temp, humidity, occupancy]], 
        columns=["Temperature", "Humidity", "Occupancy"]
    )
    
    # 2. Predict THI
    predicted_thi = model.predict(input_df)[0]
    
    # --- REALISTIC CONSTRAINTS ---
    # Even if the THI is 'mathematically' okay, extreme temps are uncomfortable/dangerous.
    if temp > 40:
        return f"{predicted_thi:.2f}", "0.0% (EXTREME HEAT DANGER 🥵)"
    if temp < 5:
        return f"{predicted_thi:.2f}", "0.0% (EXTREME COLD DANGER 🥶)"
    if humidity > 85:
        # High humidity prevents sweat evaporation, making it feel worse
        humidity_penalty = 20
    else:
        humidity_penalty = 0

    # 3. Comfort Logic based on THI
    # The 'Sweet Spot' for THI is usually between 18 and 22.
    ideal_thi = 20.0
    diff = abs(predicted_thi - ideal_thi)
    
    # We penalize more aggressively now (Multiplier of 8 instead of 5 or 6)
    comfort_pct = 100 - (diff * 8) - humidity_penalty
    
    # 4. Occupancy Penalty (Crowded rooms feel stuffy)
    if occupancy > 1:
        comfort_pct -= (occupancy * 2.5)

    # Final result cleaning
    comfort_pct = max(0, min(100, comfort_pct))
    
    # Status labels based on realistic thresholds
    if comfort_pct > 80:
        status = "Perfectly Comfortable 👌"
    elif comfort_pct > 60:
        status = "Acceptable 👍"
    elif comfort_pct > 30:
        status = "Uncomfortable ⚠️"
    else:
        status = "Highly Uncomfortable/Dangerous 🛑"
        
    return f"{predicted_thi:.2f}", f"{comfort_pct:.1f}% ({status})"

# --- 5: UI Interface Setup ---
description_text = "Enter room parameters to calculate Thermal Humidity Index (THI) and predicted human comfort levels."

interface = gr.Interface(
    fn=calculate_thermal_comfort,
    inputs=[
        gr.Number(label="Room Temperature (°C)", value=25),
        gr.Slider(0, 100, label="Humidity (%)", value=50),
        gr.Number(label="Occupancy (No. of People)", value=1)
    ],
    outputs=[
        gr.Textbox(label="Predicted THI (Index)"),
        gr.Textbox(label="Comfort Level Percentage")
    ],
    title="🏠 Smart Room Comfort Predictor",
    description=description_text
)

# --- 6: Launch ---
if __name__ == "__main__":
    print("Model trained successfully. Launching UI...")
    interface.launch(theme="soft")