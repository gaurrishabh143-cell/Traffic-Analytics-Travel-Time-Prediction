import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("dataset/traffic_data.csv")

# Weather ko numeric value mein convert karna
data["Weather"] = data["Weather"].map({
    "Clear": 0,
    "Rain": 1
})

# Features
X = data[
    [
        "Hour",
        "Traffic_Volume",
        "Average_Speed",
        "Weather",
        "Distance_km"
    ]
]

# Target
y = data["Travel_Time_min"]

# Training aur testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Machine Learning model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Model train
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("========== MODEL RESULTS ==========")

print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))

# Example prediction
new_data = pd.DataFrame({
    "Hour": [18],
    "Traffic_Volume": [2500],
    "Average_Speed": [18],
    "Weather": [1],
    "Distance_km": [10]
})

predicted_time = model.predict(new_data)

print(
    "\nPredicted Travel Time:",
    round(predicted_time[0], 2),
    "minutes"
    
)
import matplotlib.pyplot as plt

# Actual vs Predicted Travel Time
plt.figure(figsize=(10, 5))

plt.plot(
    range(len(y_test)),
    y_test.values,
    marker="o",
    label="Actual Travel Time"
)

plt.plot(
    range(len(predictions)),
    predictions,
    marker="x",
    label="Predicted Travel Time"
)

plt.title("Actual vs Predicted Travel Time")
plt.xlabel("Test Data Samples")
plt.ylabel("Travel Time (minutes)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("graphs/actual_vs_predicted.png")

plt.show()