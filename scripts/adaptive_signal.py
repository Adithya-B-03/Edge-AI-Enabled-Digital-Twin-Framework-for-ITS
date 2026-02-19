import traci
import pandas as pd
import numpy as np
import joblib
import os
import sys

# -----------------------------
# 1️⃣ Load trained ML model
# -----------------------------
model = joblib.load("models/parcl_model.pkl")

# -----------------------------
# 2️⃣ SUMO configuration
# -----------------------------
sumoCmd = [
    "sumo-gui",
    "-c", "config/map.sumo.cfg"
]

traci.start(sumoCmd)

# Your traffic light ID
TLS_ID = "256606767"

print("Simulation started...")

# -----------------------------
# 3️⃣ Simulation Loop
# -----------------------------
while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()

    # Get all vehicle IDs
    vehicle_ids = traci.vehicle.getIDList()

    if len(vehicle_ids) == 0:
        continue

    speeds = []
    accelerations = []
    co2_vals = []
    fuel_vals = []

    for vid in vehicle_ids:
        speeds.append(traci.vehicle.getSpeed(vid))
        accelerations.append(traci.vehicle.getAcceleration(vid))
        co2_vals.append(traci.vehicle.getCO2Emission(vid))
        fuel_vals.append(traci.vehicle.getFuelConsumption(vid))

    # -----------------------------
    # 4️⃣ Compute Live Features
    # -----------------------------
    vehicle_count = len(vehicle_ids)
    avg_speed = np.mean(speeds)
    speed_std = np.std(speeds)
    avg_acc = np.mean(accelerations)
    acc_std = np.std(accelerations)
    avg_co2 = np.mean(co2_vals)
    avg_fuel = np.mean(fuel_vals)
    sudden_brake_count = sum(a < -3 for a in accelerations)

    # Create feature vector
    X_live = pd.DataFrame([[
        vehicle_count,
        avg_speed,
        speed_std,
        avg_acc,
        acc_std,
        avg_co2,
        avg_fuel,
        sudden_brake_count
    ]], columns=[
        "vehicle_count",
        "avg_speed",
        "speed_std",
        "avg_acceleration",
        "acc_std",
        "avg_co2",
        "avg_fuel",
        "sudden_brake_count"
    ])

    # -----------------------------
    # 5️⃣ Predict TIS
    # -----------------------------
    predicted_TIS = model.predict(X_live)[0]

    print("Predicted TIS:", round(predicted_TIS, 3))

    # -----------------------------
    # 6️⃣ Adaptive Signal Logic
    # -----------------------------
    if predicted_TIS > 0.7:
        # High congestion → extend green
        traci.trafficlight.setPhaseDuration(TLS_ID, 40)

    elif predicted_TIS < 0.3:
        # Low congestion → reduce green
        traci.trafficlight.setPhaseDuration(TLS_ID, 15)

    else:
        # Normal traffic
        traci.trafficlight.setPhaseDuration(TLS_ID, 25)

# -----------------------------
# 7️⃣ Close Simulation
# -----------------------------
traci.close()
print("Simulation ended.")
