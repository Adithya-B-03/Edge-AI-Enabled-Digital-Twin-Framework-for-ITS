import traci
import numpy as np
import pandas as pd
import joblib

# ==========================================
# LOAD ENSEMBLE MODELS
# ==========================================
rf = joblib.load("models/rf.pkl")
gb = joblib.load("models/gb.pkl")

# ==========================================
# START SUMO
# ==========================================
sumoCmd = ["sumo-gui", "-c", "config/map.sumo.cfg"]
traci.start(sumoCmd)

# Dynamically detect traffic light ID
tls_list = traci.trafficlight.getIDList()
if len(tls_list) == 0:
    print("No traffic lights found!")
    traci.close()
    exit()

TLS_ID = tls_list[0]
print("Advanced Intelligent Controller Started")
print("Using TLS:", TLS_ID)

# ==========================================
# ENSEMBLE PREDICTION FUNCTION
# ==========================================
def predict_tis(features):
    X_live = pd.DataFrame([features], columns=[
        "vehicle_count",
        "avg_speed",
        "speed_std",
        "avg_acceleration",
        "acc_std",
        "avg_co2",
        "avg_fuel",
        "sudden_brake_count"
    ])

    tis_rf = rf.predict(X_live)[0]
    tis_gb = gb.predict(X_live)[0]

    return (tis_rf + tis_gb) / 2


# ==========================================
# MAIN CONTROL LOOP
# ==========================================
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    # ======================================
    # 🚑 PRIORITY CHECK
    # ======================================
    priority_detected = False

    for vid in traci.vehicle.getIDList():
        vtype = traci.vehicle.getTypeID(vid)
        if "emergency" in vtype.lower():
            priority_detected = True
            break

    if priority_detected:
        print("🚑 Emergency detected! Extending green.")
        traci.trafficlight.setPhaseDuration(TLS_ID, 40)
        continue

    # ======================================
    # 🚦 LANE-LEVEL TIS COMPUTATION
    # ======================================
    controlled_lanes = list(set(
        traci.trafficlight.getControlledLanes(TLS_ID)
    ))

    lane_tis = {}

    for lane in controlled_lanes:
        veh_ids = traci.lane.getLastStepVehicleIDs(lane)

        if len(veh_ids) == 0:
            lane_tis[lane] = 0
            continue

        speeds = [traci.vehicle.getSpeed(v) for v in veh_ids]
        accs = [traci.vehicle.getAcceleration(v) for v in veh_ids]
        co2 = [traci.vehicle.getCO2Emission(v) for v in veh_ids]
        fuel = [traci.vehicle.getFuelConsumption(v) for v in veh_ids]

        features = [
            len(veh_ids),
            np.mean(speeds),
            np.std(speeds),
            np.mean(accs),
            np.std(accs),
            np.mean(co2),
            np.mean(fuel),
            sum(a < -3 for a in accs)
        ]

        lane_tis[lane] = predict_tis(features)

    # ======================================
    # 📊 PHASE-LEVEL UTILITY COMPUTATION
    # ======================================
    logics = traci.trafficlight.getAllProgramLogics(TLS_ID)
    phases = logics[0].phases

    phase_utilities = []

    controlled_lanes = traci.trafficlight.getControlledLanes(TLS_ID)

    for phase_index, phase in enumerate(phases):
        state = phase.state  # e.g., "GrGr"
        utility = 0

        for lane_index, signal in enumerate(state):
            if signal.lower() == 'g':  # green signal
                lane = controlled_lanes[lane_index]
                utility += lane_tis.get(lane, 0)

        phase_utilities.append(utility)

    # Select best phase
    best_phase = int(np.argmax(phase_utilities))
    best_utility = phase_utilities[best_phase]

    # ======================================
    # 🚦 APPLY BEST PHASE
    # ======================================
    traci.trafficlight.setPhase(TLS_ID, best_phase)

    green_time = 15 + int(30 * best_utility)
    traci.trafficlight.setPhaseDuration(TLS_ID, green_time)

    print("Selected Phase:", best_phase,
          "| Utility:", round(best_utility, 3),
          "| Green:", green_time)

# ==========================================
# END SIMULATION
# ==========================================
traci.close()
print("Simulation ended.")