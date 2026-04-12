# 🔥 Smart Thermal Comfort Predictor

A machine learning-based system that predicts indoor thermal comfort using environmental parameters like temperature, humidity, CO₂, light, and humidity ratio. The system provides real-time comfort analysis, air quality status, and intelligent cooling & ventilation suggestions.

---

## 🚀 Live Demo

🌐 **Web App:**
https://ajay-kumar90-thermal-comfort-predictor.hf.space

📱 **Mobile App:**
Android APK available (WebView-based app)

---

## 📌 Features

* ✅ Thermal Comfort Prediction (THI based)
* ✅ Machine Learning Model (Random Forest)
* ✅ Comfort Percentage Score (0–100%)
* ✅ Air Quality Detection (CO₂ based)
* ✅ Ventilation Suggestions
* ✅ AC / Cooling Recommendation System
* ✅ Graph Visualization (Temperature vs THI)
* ✅ Real-time User Input Interface
* ✅ Web Deployment using Gradio
* ✅ Mobile App (Android WebView)

---

## 🧠 Technologies Used

* **Python**

  * Pandas
  * NumPy
  * Scikit-learn
  * Matplotlib
  * Gradio

* **Machine Learning**

  * Random Forest Regressor
  * Random Forest Classifier

* **Deployment**

  * Hugging Face Spaces

* **Mobile App**

  * Android Studio (Java + WebView)

---

## 📊 Input Parameters

* 🌡️ Temperature (°C)
* 💧 Humidity (%)
* 💡 Light
* 🫁 CO₂ (ppm)
* 💨 Humidity Ratio

---

## 📈 Output

* 📊 Predicted THI
* 🎯 Comfort Percentage
* 😊 Comfort Status (Comfortable / Uncomfortable)
* ❄️ AC Recommendation
* 🌬️ Ventilation Suggestion
* 🟢 Air Quality Indicator
* 📉 Graph (Temperature vs THI)

---

## 🧮 THI Formula

```
THI = T - (0.55 - 0.0055 × RH) × (T - 14.5)
```

Where:

* T = Temperature (°C)
* RH = Relative Humidity (%)

---

## 🤖 Machine Learning Approach

* Data cleaning and preprocessing
* Feature engineering (hour, day)
* Regression model → THI prediction
* Classification model → Comfort detection
* Hybrid logic → Real-world constraints applied

---

## ⚠️ Smart Rules (Real-world Constraints)

* Extreme Heat (>40°C) → Dangerous
* Extreme Cold (<10°C) → Uncomfortable
* High CO₂ (>1500 ppm) → Poor Air Quality
* High Humidity (>70%) → Reduced comfort

---

## 📱 Mobile Application

* Built using Android Studio
* Uses WebView to load live web app
* Lightweight and easy to use

---

## 🎓 Project Use Cases

* Smart Homes 🏠
* Offices 🏢
* Classrooms 🎓
* HVAC Systems
* Indoor Environment Monitoring

---

## 🔮 Future Scope

* IoT sensor integration
* Smart AC automation
* Mobile native UI improvements
* Real-time sensor data collection

---

## 👨‍💻 Author

**Ajay Kumar**
MCA Student
K.R. Mangalam University

---

## 📜 License

This project is for educational and research purposes.

---

## 🙌 Acknowledgment

This project is developed as part of a final year research project focusing on indoor thermal comfort using machine learning.

---

