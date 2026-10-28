import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Project paths
DATA_PATH = os.path.join("Data", "IMDB Movies India.csv")
OUTPUT_DIR = "Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading movie dataset...")

# Load dataset
df = pd.read_csv(DATA_PATH, encoding="latin1")
df.columns = df.columns.str.strip()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Clean column names and identify target
rating_column = next(
    (col for col in df.columns if col.lower() == "rating"),
    None
)

if rating_column is None:
    raise ValueError(
        "Rating column not found. Please check the CSV column names."
    )

# Keep useful features that may exist in the dataset
possible_features = [
    "Genre", "Director", "Actor 1", "Actor 2", "Actor 3",
    "Year", "Duration", "Votes"
]
feature_columns = [col for col in possible_features if col in df.columns]

if not feature_columns:
    raise ValueError("No expected feature columns found in the dataset.")

# Prepare target and features
df[rating_column] = pd.to_numeric(df[rating_column], errors="coerce")
df = df.dropna(subset=[rating_column]).copy()

# Convert numeric-like columns
for col in ["Year", "Duration", "Votes"]:
    if col in df.columns:
        cleaned = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(\d+\.?\d*)", expand=False)
        )
        df[col] = pd.to_numeric(cleaned, errors="coerce")

X = df[feature_columns].copy()
y = df[rating_column].copy()

# Remove rows where target is outside IMDb's usual rating scale
valid = y.between(0, 10)
X = X.loc[valid]
y = y.loc[valid]

# Detect numeric and categorical features
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = [
    col for col in X.columns if col not in numeric_features
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining movie rating prediction model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("----------------")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R-squared (R2): {r2:.3f}")

# Save evaluation report
with open(os.path.join(OUTPUT_DIR, "evaluation.txt"), "w",
          encoding="utf-8") as file:
    file.write("Movie Rating Prediction - Evaluation\n")
    file.write(f"Rows used: {len(X)}\n")
    file.write(f"Features: {feature_columns}\n")
    file.write(f"MAE: {mae:.3f}\n")
    file.write(f"RMSE: {rmse:.3f}\n")
    file.write(f"R2: {r2:.3f}\n")

# Save actual vs predicted plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=predictions, alpha=0.5)
plt.xlabel("Actual Movie Rating")
plt.ylabel("Predicted Movie Rating")
plt.title("Actual vs Predicted Movie Ratings")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"),
    dpi=150
)
plt.close()

# Save trained model
joblib.dump(model, os.path.join(OUTPUT_DIR, "movie_rating_model.joblib"))

# Save a sample of predictions
results = pd.DataFrame({
    "Actual_Rating": y_test.to_numpy(),
    "Predicted_Rating": predictions
})
results.to_csv(
    os.path.join(OUTPUT_DIR, "predictions.csv"),
    index=False
)

print("\nAll done! Model and results saved in the Output folder.")