from fastapi import FastAPI, HTTPException
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
    load_type: str  # e.g., "residential", "commercial", "industrial"

@app.post("/ingest")
def ingest(measurement: Measurement):
    try:
        point = Point("power") \
            .tag("device", measurement.device_id) \
            .tag("load_type", measurement.load_type) \
            .field("p_active", measurement.power) \
            .time(time.time_ns())

        write_api.write(bucket=bucket, record=point)
        return {
            "status": "ok",
            "device": measurement.device_id,
            "power": measurement.power,
            "load_type": measurement.load_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest data: {str(e)}")
