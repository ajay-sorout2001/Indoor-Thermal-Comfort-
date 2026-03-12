
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# 1 Load Dataset


data = pd.read_csv("thermal_data.csv")

print("Dataset Preview")
print(data.head())

print("\nDataset Info")
print(data.info())


# 2 Data Cleaning


# Remove duplicates
data.drop_duplicates(inplace=True)

# Fill missing numeric values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Convert columns to numeric safely
for col in data.columns:
    try:
        data[col] = pd.to_numeric(data[col])
    except:
        pass

print("\nCleaned Dataset")
print(data.head())


# 3 Calculate THI


data["THI"] = data["Temperature"] - (0.55 - 0.0055 * data["Humidity"]) * (data["Temperature"] - 14.5)

print("\nDataset with THI")
print(data.head())


# 4 Visualization


# Scatter Plot
plt.figure()
sns.scatterplot(x=data["Temperature"], y=data["THI"])
plt.title("Temperature vs THI")
plt.xlabel("Temperature")
plt.ylabel("THI")
plt.show()

# Correlation Heatmap
numeric_data = data.select_dtypes(include=['float64','int64'])

corr_matrix = numeric_data.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()


# 5 Machine Learning Model


X = data[["Temperature","Humidity","Occupancy"]]
y = data["THI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)



# 6 Model Evaluation


print("\nModel Evaluation")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)



# 7 Feature Importance (Important Graph)


importance = model.feature_importances_

features = X.columns

plt.figure()
plt.bar(features, importance)

plt.title("Feature Importance (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.show()


# 8 Prediction Example

new_data = pd.DataFrame(
    [[27,65,10]],
    columns=["Temperature","Humidity","Occupancy"]
)

prediction = model.predict(new_data)

print("\nPredicted THI:", prediction)
