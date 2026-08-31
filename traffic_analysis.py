import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("dataset/traffic_data.csv")

# Convert Weather into numbers
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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Linear Regression
# -----------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_mse = mean_squared_error(
    y_test,
    linear_predictions
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


# -----------------------------
# Random Forest
# -----------------------------

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_mse = mean_squared_error(
    y_test,
    rf_predictions
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


# -----------------------------
# Print Results
# -----------------------------

print("========== MODEL COMPARISON ==========")

print("\nLinear Regression")
print("MAE:", round(linear_mae, 2))
print("MSE:", round(linear_mse, 2))
print("R2 Score:", round(linear_r2, 2))

print("\nRandom Forest Regression")
print("MAE:", round(rf_mae, 2))
print("MSE:", round(rf_mse, 2))
print("R2 Score:", round(rf_r2, 2))


# -----------------------------
# Select Best Model
# -----------------------------

if rf_r2 > linear_r2:
    best_model = random_forest
    best_predictions = rf_predictions
    best_model_name = "Random Forest Regression"
else:
    best_model = linear_model
    best_predictions = linear_predictions
    best_model_name = "Linear Regression"


print("\nBest Model:", best_model_name)


# -----------------------------
# Actual vs Predicted Graph
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    range(len(y_test)),
    y_test.values,
    marker="o",
    label="Actual"
)

plt.plot(
    range(len(best_predictions)),
    best_predictions,
    marker="x",
    label="Predicted"
)

plt.title("Actual vs Predicted Travel Time")
plt.xlabel("Test Data Samples")
plt.ylabel("Travel Time (minutes)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("graphs/actual_vs_predicted.png")

plt.close()


# -----------------------------
# Example Prediction
# -----------------------------

new_data = pd.DataFrame({
    "Hour": [18],
    "Traffic_Volume": [2500],
    "Average_Speed": [18],
    "Weather": [1],
    "Distance_km": [10]
})

predicted_time = best_model.predict(new_data)

print(
    "\nExample Prediction:"
)

print(
    "Predicted Travel Time:",
    round(predicted_time[0], 2),
    "minutes"
)

print("\n========== PROJECT COMPLETE ==========")