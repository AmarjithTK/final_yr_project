import requests, random, time

API="http://localhost:8000/ingest"
DEVICES=["home1","home2","home3","home4","home5"]

while True:
    for d in DEVICES:
        payload={"device_id": d, "power": random.uniform(100, 1000)}
        r=requests.post(API, json=payload)
        print("Saving data from", d, "→", r.json())
    time.sleep(2)
