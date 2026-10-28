import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project folders
DATA_PATH = "Data/Advertising.csv"
OUTPUT_DIR = "Output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove an unnecessary index column if present
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Check required columns
required_columns = ["TV", "Radio", "Newspaper", "Sales"]

if not all(column in df.columns for column in required_columns):
    raise ValueError(
        f"Dataset columns are incorrect. Found: {list(df.columns)}"
    )

# Select input features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

# Save evaluation results
with open(os.path.join(OUTPUT_DIR, "evaluation.txt"), "w") as file:
    file.write("SALES PREDICTION MODEL EVALUATION\n")
    file.write("---------------------------------\n")
    file.write(f"Dataset rows: {len(df)}\n")
    file.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    file.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    file.write(f"R-squared (R2 Score): {r2:.4f}\n")

# Save predictions
predictions = pd.DataFrame({
    "Actual Sales": y_test.to_numpy(),
    "Predicted Sales": y_pred
})
predictions.to_csv(
    os.path.join(OUTPUT_DIR, "predictions.csv"), index=False
)

# Save trained model
joblib.dump(model, os.path.join(OUTPUT_DIR, "sales_model.joblib"))

# Plot actual vs predicted sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"))
plt.close()

print("Model training completed successfully!")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Results saved in the '{OUTPUT_DIR}' folder.")