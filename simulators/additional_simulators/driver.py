import importlib
import requests
import time
import sys

API = "http://localhost:8000/ingest"

# Define simulator mapping: sim_name → module_name
SIMULATOR_MAP = {
    "homeload1": "homeload",
    "homeload2": "homeload",
    "homeload3": "homeload",
    "industrial1": "industrial",
    "industrial2": "industrial",
    "commercial1": "commercial",
    "commercial2": "commercial",
    # ...add more as needed...
}

DEFAULT_SIMULATORS = ["homeload1", "homeload2", "homeload3"]
DEFAULT_FREQ_SEC = 120  # 2 minutes

def run_simulator(sim_name, freq_sec):
    if sim_name not in SIMULATOR_MAP:
        print(f"Simulator '{sim_name}' not defined in SIMULATOR_MAP.")
        return
    module_name = SIMULATOR_MAP[sim_name]
    sim_module = importlib.import_module(module_name)
    for payload in sim_module.generate_data():
        r = requests.post(API, json=payload)
        print(f"Saving data from {payload['device_id']} → {r.json()}")
        time.sleep(freq_sec)

if __name__ == "__main__":
    # Usage: python driver.py [sim1 sim2 ...] [--freq seconds]
    args = sys.argv[1:]
    freq_sec = DEFAULT_FREQ_SEC
    sims = []

    # Parse --freq argument if present
    if "--freq" in args:
        idx = args.index("--freq")
        if idx + 1 < len(args):
            freq_sec = int(float(args[idx + 1]))
            args = args[:idx] + args[idx+2:]
    if args:
        sims = args
    else:
        sims = DEFAULT_SIMULATORS

    for sim_name in sims:
        run_simulator(sim_name, freq_sec)
