from fastapi import FastAPI
from pydantic import BaseModel
from influxdb_client import InfluxDBClient, Point
import os, time

app = FastAPI()

# Influx setup
client = InfluxDBClient(
    url=os.getenv("INFLUX_URL", "http://localhost:8086"),
    token=os.getenv("INFLUX_TOKEN", "admin123"),
    org=os.getenv("INFLUX_ORG", "myorg")
)
write_api = client.write_api()
bucket = os.getenv("INFLUX_BUCKET", "energy")

class Measurement(BaseModel):
    device_id: str
    power: float

@app.post("/ingest")
def ingest(measurement: Measurement):
    point = Point("power") \
        .tag("device", measurement.device_id) \
        .field("p_active", measurement.power) \
        .time(time.time_ns())

    write_api.write(bucket=bucket, record=point)
    return {"status": "ok", "device": measurement.device_id, "power": measurement.power}
