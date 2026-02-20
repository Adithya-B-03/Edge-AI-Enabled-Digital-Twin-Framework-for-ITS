import traci
import numpy as np

sumoCmd = ["sumo-gui", "-c", "config/map.sumo.cfg"]
traci.start(sumoCmd)

tls_list = traci.trafficlight.getIDList()
TLS_ID = tls_list[0]

print("Running Fixed-Time Controller")

waiting_times = []
co2_list = []
fuel_list = []
queue_lengths = []

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()

    step_wait = 0
    step_co2 = 0
    step_fuel = 0

    for v in vehicles:
        step_wait += traci.vehicle.getWaitingTime(v)
        step_co2 += traci.vehicle.getCO2Emission(v)
        step_fuel += traci.vehicle.getFuelConsumption(v)

    waiting_times.append(step_wait)
    co2_list.append(step_co2)
    fuel_list.append(step_fuel)

    lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
    queue_lengths.append(queue)

traci.close()

print("\n===== FIXED TIME RESULTS =====")
print("Average Waiting Time:", np.mean(waiting_times))
print("Average Queue Length:", np.mean(queue_lengths))
print("Average CO2:", np.mean(co2_list))
print("Average Fuel:", np.mean(fuel_list))