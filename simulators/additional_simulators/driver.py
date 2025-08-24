import importlib
import requests
import time
import sys

API = "http://localhost:8000/ingest"

# List of simulator module names (without .py)
SIMULATORS = ["homeload", "industrial", "commercial"]

def run_simulator(sim_name):
    sim_module = importlib.import_module(sim_name)
    for payload in sim_module.generate_data():
        r = requests.post(API, json=payload)
        print(f"Saving data from {payload['device_id']} → {r.json()}")
        time.sleep(2)

if __name__ == "__main__":
    # Optionally, accept simulator names as command-line arguments
    sims = sys.argv[1:] if len(sys.argv) > 1 else SIMULATORS
    for sim_name in sims:
        run_simulator(sim_name)
