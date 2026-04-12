import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

# --- LOAD DATA ---
data1 = pd.read_csv("DataTest.csv")
data2 = pd.read_csv("datatest1.csv")

data = pd.concat([data1, data2], ignore_index=True)

# --- CLEAN ---
data.drop_duplicates(inplace=True)

data['date'] = pd.to_datetime(data['date'], dayfirst=True, errors='coerce')
data.dropna(subset=['date'], inplace=True)

data.fillna(data.mean(numeric_only=True), inplace=True)

# --- FEATURES ---
data['hour'] = data['date'].dt.hour
data['day'] = data['date'].dt.day

# --- THI ---
data["THI"] = data["Temperature"] - (0.55 - 0.0055 * data["Humidity"]) * (data["Temperature"] - 14.5)

# --- LABEL ---
data["Comfort_Label"] = data["THI"].apply(lambda x: 1 if 18 <= x <= 24 else 0)

features = ["Temperature","Humidity","Light","CO2","HumidityRatio","hour","day"]

X = data[features]
y_reg = data["THI"]
y_clf = data["Comfort_Label"]

# --- SPLIT ---
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# --- MODEL ---
reg_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
clf_model = RandomForestClassifier(n_estimators=200, random_state=42)

reg_model.fit(X_train, y_reg_train)
clf_model.fit(X_train, y_clf_train)

# --- FUNCTION ---
def predict_comfort(temp, humidity, light, co2, humidity_ratio):

    from datetime import datetime
    now = datetime.now()
    hour = now.hour
    day = now.day

    if humidity_ratio > 1:
        humidity_ratio = humidity_ratio / 10000

    input_df = pd.DataFrame(
        [[temp, humidity, light, co2, humidity_ratio, hour, day]],
        columns=features
    )

    thi = reg_model.predict(input_df)[0]

    # 🚨 EXTREME CONDITIONS
    if temp >= 40:
        return f"{thi:.2f}", "0%", "❌ Extreme Heat 🥵", "❄️ Max Cooling", "🔴 Poor", "Open windows immediately", None

    if temp <= 10:
        return f"{thi:.2f}", "0%", "❌ Extreme Cold 🥶", "🔥 Heating", "🔴 Poor", "Close windows", None

    # --- COMFORT SCORE ---
    score = 100 - abs(thi - 21) * 7

    if humidity > 70:
        score -= 15
    if co2 > 1000:
        score -= 10
    if temp > 30 or temp < 18:
        score -= 25

    score = max(0, min(100, score))

    # --- STATUS ---
    if score > 80:
        status = "✅ Comfortable"
    elif score > 60:
        status = "🙂 Acceptable"
    elif score > 30:
        status = "⚠️ Uncomfortable"
    else:
        status = "❌ Bad"

    # --- AC ---
    if temp > 30:
        ac = "❄️ High Cooling"
    elif temp > 26:
        ac = "🌬️ Medium Cooling"
    elif temp > 22:
        ac = "🌀 Fan enough"
    else:
        ac = "🛑 No Cooling"

    # --- AIR QUALITY ---
    if co2 < 800:
        air = "🟢 Good"
    elif co2 < 1200:
        air = "🟡 Moderate"
    else:
        air = "🔴 Poor"

    # --- VENTILATION ---
    if co2 > 1500:
        ventilation = "🪟 Open windows + exhaust"
    elif co2 > 1000:
        ventilation = "🌬️ Use fan"
    else:
        ventilation = "✅ Air OK"

    # --- GRAPH ---
    temps = np.linspace(15, 40, 50)
    thi_vals = temps - (0.55 - 0.0055 * humidity) * (temps - 14.5)

    plt.figure()
    plt.plot(temps, thi_vals)
    plt.scatter(temp, thi, color='red')
    plt.title("Temperature vs THI")

    return f"{thi:.2f}", f"{score:.1f}%", status, ac, air, ventilation, plt

# --- UI ---
interface = gr.Interface(
    fn=predict_comfort,
    inputs=[
        gr.Number(label="Temperature", value=25),
        gr.Slider(0,100,label="Humidity", value=50),
        gr.Number(label="Light", value=300),
        gr.Number(label="CO2", value=500),
        gr.Number(label="Humidity Ratio", value=0.004)
    ],
    outputs=[
        gr.Textbox(label="THI"),
        gr.Textbox(label="Comfort %"),
        gr.Textbox(label="Status"),
        gr.Textbox(label="AC"),
        gr.Textbox(label="Air Quality"),
        gr.Textbox(label="Ventilation"),
        gr.Plot(label="Graph")
    ]
)

interface.launch()
