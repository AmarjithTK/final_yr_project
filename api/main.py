from fastapi import FastAPI
from pydantic import BaseModel
from influxdb_client import InfluxDBClient, Point
import os, time
import logging

app = FastAPI()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Starting FastAPI application.")

# Influx setup
logger.info("Initializing InfluxDBClient with URL: %s", os.getenv("INFLUX_URL", "http://influxdb:8086"))
client = InfluxDBClient(
    url=os.getenv("INFLUX_URL", "http://influxdb:8086"),
    token=os.getenv("INFLUX_TOKEN", "my-super-secret-auth-token"),
    org=os.getenv("INFLUX_ORG", "myorg")
)

write_api = client.write_api()
bucket = os.getenv("INFLUX_BUCKET", "energy")
logger.info("Using InfluxDB bucket: %s", bucket)

class Measurement(BaseModel):
    device_id: str
    power: float

@app.on_event("startup")
def on_startup():
    logger.info("FastAPI application startup event triggered.")

@app.post("/ingest")
def ingest(measurement: Measurement):
    logger.info("Received /ingest request: device_id=%s, power=%s", measurement.device_id, measurement.power)
    try:
        point = Point("power") \
            .tag("device", measurement.device_id) \
            .field("p_active", measurement.power) \
            .time(time.time_ns())
        logger.debug("Constructed InfluxDB Point: %s", point.to_line_protocol())
        write_api.write(bucket=bucket, record=point)
        logger.info("Successfully wrote data to InfluxDB for device_id=%s", measurement.device_id)
        return {"status": "oksdgk", "device": measurement.device_id, "power": measurement.power}
    except Exception as e:
        logger.error("Failed to write to InfluxDB: %s", str(e), exc_info=True)
        return {"status": "error", "error": str(e)}

