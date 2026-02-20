import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("output/intelligent_dataset.csv")

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 1 - Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Model 2 - Gradient Boosting
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)

print("RF R2:", r2_score(y_test, rf.predict(X_test)))
print("GB R2:", r2_score(y_test, gb.predict(X_test)))

# Save models
joblib.dump(rf, "models/rf.pkl")
joblib.dump(gb, "models/gb.pkl")

print("Ensemble models saved successfully.")