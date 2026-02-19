import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -----------------------------
# 1️⃣ Load intelligent dataset
# -----------------------------
df = pd.read_csv("output/intelligent_dataset.csv")

print("Dataset Loaded")
print("Shape:", df.shape)

# -----------------------------
# 2️⃣ Define Features (X) and Target (y)
# -----------------------------
X = df[[
    "vehicle_count",
    "avg_speed",
    "speed_std",
    "avg_acceleration",
    "acc_std",
    "avg_co2",
    "avg_fuel",
    "sudden_brake_count"
]]

y = df["TIS"]

# -----------------------------
# 3️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------
# 4️⃣ Initialize Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# -----------------------------
# 5️⃣ Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 6️⃣ Evaluate Model
# -----------------------------
pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\nModel Performance:")
print("MSE:", mse)
print("R2 Score:", r2)

# -----------------------------
# 7️⃣ Save Model
# -----------------------------
joblib.dump(model, "models/parcl_model.pkl")

print("\nModel saved successfully as parcl_model.pkl")
