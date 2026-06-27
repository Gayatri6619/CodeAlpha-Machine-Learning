import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Get the folder where this script is saved
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create the full path to the CSV file
csv_path = os.path.join(current_dir, "Advertising.csv")

# Load dataset
data = pd.read_csv(csv_path)

# Features and target
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))