import requests, random, time, argparse

API = "http://apifinal.atherpulse.in/ingest"
DEVICES = ["home1", "home2", "home3", "home4", "home5"]

def simulate(interval):
    while True:
        for d in DEVICES:
            payload = {
                "device_id": d,
                "voltage": random.uniform(210, 240),    # V
                "current": random.uniform(1, 10),      # I
                "power": random.uniform(100, 1000),    # P
                "cos_phi": random.uniform(0.8, 1.0)    # PF
            }
            try:
                r = requests.post(API, json=payload, timeout=5)
                print(f"Saving data from {d} → {r.status_code}")
            except Exception as e:
                print(f"Error sending from {d}: {e}")
        time.sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interval", 
        type=int, 
        default=900,   # 900 sec = 15 min
        help="Interval in seconds between posts (default: 900)"
    )
    args = parser.parse_args()
    simulate(args.interval)
