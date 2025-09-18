import requests
import random
import time
import sqlite3
import json
import os
from datetime import datetime, timedelta

API_URL = "https://apifinal.atherpulse.in/fulldatatest"
DB_PATH = "simulator_data.db"
CONFIG_PATH = "sim_config.json"

DEVICE_TYPES = ["commercial", "industrial", "residential"]
NODES = ["node1", "node2", "node3"]

DEVICE_LIMITS = {
    "commercial": {"v": (210, 250), "i": (0, 100), "p": (0, 20000), "q": (0, 10000)},
    "industrial": {"v": (210, 250), "i": (0, 500), "p": (0, 100000), "q": (0, 50000)},
    "residential": {"v": (210, 250), "i": (0, 50), "p": (0, 10000), "q": (0, 5000)},
}

def setup_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS aggregates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        node TEXT,
        device_type TEXT,
        timestamp TEXT,
        v REAL,
        i REAL,
        p REAL,
        q REAL,
        kwh REAL
    )''')
    conn.commit()
    conn.close()

def load_or_create_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    else:
        # Generate random node_types and save
        node_types = {node: random.choice(DEVICE_TYPES) for node in NODES}
        config = {"node_types": node_types}
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
    return config

def store_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def store_aggregate(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO aggregates (node, device_type, timestamp, v, i, p, q, kwh)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (data["node"], data["device_type"], data["timestamp"], data["v"], data["i"], data["p"], data["q"], data["kwh"]))
    conn.commit()
    conn.close()

def send_verification():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    today = datetime.now().date().isoformat()
    c.execute("SELECT * FROM aggregates WHERE timestamp LIKE ?", (f"{today}%",))
    rows = c.fetchall()
    conn.close()
    # Send all aggregates for today as verification
    for row in rows:
        payload = {
            "node": row[1],
            "device_type": row[2],
            "timestamp": row[3],
            "v": row[4],
            "i": row[5],
            "p": row[6],
            "q": row[7],
            "kwh": row[8]
        }
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            print(f"Verification Sent: {payload} | Response: {response.status_code}")
        except Exception as e:
            print(f"Verification Error: {e}")

if __name__ == "__main__":
    setup_db()
    config = load_or_create_config()
    node_types = config["node_types"]
    while True:
        # If you want to randomize node_types each day, uncomment below:
        # node_types = {node: random.choice(DEVICE_TYPES) for node in NODES}
        # config["node_types"] = node_types
        # store_config(config)

        node_totals = {node: {"p": 0, "q": 0} for node in NODES}
        v_vals = {node: round(random.uniform(*DEVICE_LIMITS[node_types[node]]["v"]), 2) for node in NODES}
        i_vals = {node: round(random.uniform(*DEVICE_LIMITS[node_types[node]]["i"]), 2) for node in NODES}
        total_seconds = 15 * 60
        total_seconds =2
        for _ in range(total_seconds):
            for node in NODES:
                limits = DEVICE_LIMITS[node_types[node]]
                p_inst = random.uniform(*limits["p"])
                q_inst = random.uniform(*limits["q"])
                node_totals[node]["p"] += p_inst
                node_totals[node]["q"] += q_inst
            time.sleep(1)
        timestamp = datetime.now().isoformat()
        for node in NODES:
            limits = DEVICE_LIMITS[node_types[node]]
            avg_p = node_totals[node]["p"] / total_seconds
            avg_q = node_totals[node]["q"] / total_seconds
            kwh = node_totals[node]["p"] / 3600000
            agg = {
                "node": node,
                "device_type": node_types[node],
                "timestamp": timestamp,
                "v": v_vals[node],
                "i": i_vals[node],
                "p": round(avg_p, 2),
                "q": round(avg_q, 2),
                "kwh": round(kwh, 4)
            }
            store_aggregate(agg)
            print(f"Aggregated 15min for {node}: {agg}")
            try:
                response = requests.post(API_URL, json=agg, timeout=5)
                print(f"Sent: {agg} | Response: {response.status_code}")
            except Exception as e:
                print(f"Error sending data: {e}")
        # If it's end of day (23:59), send verification
        now = datetime.now()
        if now.hour == 23 and now.minute >= 50:
            send_verification()
        # Wait until next 15min slot
        next_slot = (now + timedelta(minutes=15)).replace(second=0, microsecond=0)
        sleep_time = (next_slot - datetime.now()).total_seconds()
        time.sleep(max(sleep_time, 0))