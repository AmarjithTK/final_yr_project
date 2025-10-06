import requests
import random
import time



# changes required

proper 15 minutes time, 
aggregating power instantaneous each second for 15 minutes
storing total data in some database in proper format, may use sqlite
sending verification request at the end of the day to overlap with the existing data


API_URL = "https://apifinal.atherpulse.in/fulldatatest"

DEVICE_TYPES = ["commercial", "industrial", "residential"]
NODES = ["node1", "node2", "node3"]

# Define suitable limits for each device type
DEVICE_LIMITS = {
    "commercial": {
        "v": (210, 250),
        "i": (0, 100),
        "p": (0, 20000),
        "q": (0, 10000),
        "kwh": (0, 5000)
    },
    "industrial": {
        "v": (210, 250),
        "i": (0, 500),
        "p": (0, 100000),
        "q": (0, 50000),
        "kwh": (0, 20000)
    },
    "residential": {
        "v": (210, 250),
        "i": (0, 50),
        "p": (0, 10000),
        "q": (0, 5000),
        "kwh": (0, 1000)
    }
}

def generate_payload():
    device_type = random.choice(DEVICE_TYPES)
    limits = DEVICE_LIMITS[device_type]
    return {
        "device_id": f"sim-{random.randint(1, 10)}",
        "v": round(random.uniform(*limits["v"]), 2),
        "i": round(random.uniform(*limits["i"]), 2),
        "p": round(random.uniform(*limits["p"]), 2),
        "q": round(random.uniform(*limits["q"]), 2),
        "device_type": device_type,
        "node": random.choice(NODES),
        "kwh": round(random.uniform(*limits["kwh"]), 2)
    }

if __name__ == "__main__":
    while True:
        payload = generate_payload()
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            print(f"Sent: {payload} | Response: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Error sending data: {e}")
        time.sleep(10)
