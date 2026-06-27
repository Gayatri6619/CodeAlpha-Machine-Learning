import pandas as pd

# Load dataset
df = pd.read_csv("car data.csv")

# Display first 5 rows
print(df.head())
# Display first 5 rows
print(df.head())

# Number of rows and columns
print("\nShape of the dataset:")
print(df.shape)

# Column names and data types
print("\nDataset Information:")
print(df.info())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Check for duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicate rows (if any)
df.drop_duplicates(inplace=True)

# Create a new feature: Car Age
df["Car_Age"] = 2025 - df["Year"]

# Remove unnecessary columns
df.drop(["Car_Name", "Year"], axis=1, inplace=True)

# Convert categorical columns into numeric values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["Fuel_Type"] = le.fit_transform(df["Fuel_Type"])
df["Selling_type"] = le.fit_transform(df["Selling_type"])
df["Transmission"] = le.fit_transform(df["Transmission"])

# Display the updated dataset
print(df.head())
from sklearn.model_selection import train_test_split

# Features (X) and Target (y)
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.ensemble import RandomForestRegressor

# Create the model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

print("Model trained successfully!")
# Predict on test data
y_pred = model.predict(X_test)

print("Predicted Prices:")
print(y_pred[:10])  # Show first 10 predictions
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)
import matplotlib.pyplot as plt

importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.bar(features, importance)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()