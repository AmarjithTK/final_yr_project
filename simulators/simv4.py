import requests
import random
import time
import sqlite3
import json
import os
from datetime import datetime, timedelta, timezone

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

def get_offset(node):
    # Assign offsets so ranges don't overlap
    offsets = {
        "node1": 0,      # residential
        "node2": 12000,  # commercial
        "node3": 60000   # industrial
    }
    return offsets[node]

def load_or_create_config():
    # Explicitly set node types
    node_types = {
        "node1": "residential",
        "node2": "commercial",
        "node3": "industrial"
    }
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
        node_totals = {node: {"p": 0, "q": 0} for node in NODES}
        v_vals = {node: round(random.uniform(*DEVICE_LIMITS[node_types[node]]["v"]), 2) for node in NODES}
        i_vals = {node: round(random.uniform(*DEVICE_LIMITS[node_types[node]]["i"]), 2) for node in NODES}
        total_seconds = 2  # For testing

        utc_now = datetime.now(timezone.utc).replace(microsecond=0)
        timestamp = utc_now.isoformat().replace("+00:00", "Z")

        for _ in range(total_seconds):
            for node in NODES:
                limits = DEVICE_LIMITS[node_types[node]]
                offset = get_offset(node)
                p_inst = random.uniform(*limits["p"]) + offset
                q_inst = random.uniform(*limits["q"]) + offset // 2
                node_totals[node]["p"] += p_inst
                node_totals[node]["q"] += q_inst
            time.sleep(1)

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

        now = datetime.now()
        if now.hour == 23 and now.minute == 50:
            send_verification()
        next_slot = (now + timedelta(minutes=15)).replace(second=0, microsecond=0)
        sleep_time = (next_slot - datetime.now()).total_seconds()
        time.sleep(2)



# penalty conditions
# in domen=stic no penalty 
#  pf < 095 pentality
# pf < 090 penalty 1
# pf < 085 penalty 2
# pf < 080 penalty 3

# penalty for residential 

#weekend and weekdays graph
#monthly and yearly graph


#6-7 slides


#, I , cospi

#power factor measurement board


#flow chart

#prediction flowchart

#errors and corrections we need to use

#sinewave to square wave


#real power , reactive powe 

#these

