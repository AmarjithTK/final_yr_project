from fastapi import FastAPI
from pydantic import BaseModel
from influxdb_client import InfluxDBClient, Point
import os, time
import logging
from typing import Literal, Optional

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
bucket = os.getenv("INFLUX_BUCKET", "energy3")
logger.info("Using InfluxDB bucket: %s", bucket)

class Measurement(BaseModel):
    device_id: str
    power: float
    timestamp: Optional[str] = None  # ISO format string

class FullDataTest(BaseModel):
    node: str
    device_type: Literal['commercial', 'industrial', 'residential']
    timestamp: Optional[str] = None  # ISO format string
    v: float
    i: float
    p: float
    q: float
    kwh: float

def iso_to_ns(iso_str: str) -> int:
    from datetime import datetime
    import dateutil.parser
    dt = dateutil.parser.isoparse(iso_str)
    return int(dt.timestamp() * 1e9)

@app.on_event("startup")
def on_startup():
    logger.info("FastAPI application startup event triggered.")

@app.post("/ingest")
def ingest(measurement: Measurement):
    logger.info("Received /ingest request: device_id=%s, power=%s, timestamp=%s", measurement.device_id, measurement.power, measurement.timestamp)
    try:
        if measurement.timestamp:
            point_time = iso_to_ns(measurement.timestamp)
        else:
            point_time = time.time_ns()
        point = Point("power") \
            .tag("device", measurement.device_id) \
            .field("p_active", measurement.power) \
            .time(point_time)
        logger.debug("Constructed InfluxDB Point: %s", point.to_line_protocol())
        write_api.write(bucket=bucket, record=point)
        logger.info("Successfully wrote data to InfluxDB for device_id=%s", measurement.device_id)
        return {"status": "ok", "device": measurement.device_id, "power": measurement.power}
    except Exception as e:
        logger.error("Failed to write to InfluxDB: %s", str(e), exc_info=True)
        return {"status": "error", "error": str(e)}

@app.post("/fulldatatest")
def fulldatatest(data: FullDataTest):
    logger.info(
        "Received /fulldatatest request: node=%s, device_type=%s, timestamp=%s, v=%s, i=%s, p=%s, q=%s, kwh=%s",
        data.node, data.device_type, data.timestamp, data.v, data.i, data.p, data.q, data.kwh
    )
    try:
        if data.timestamp:
            point_time = iso_to_ns(data.timestamp)
        else:
            point_time = time.time_ns()
        point = (
            Point("fulldatatestv2")
            .tag("device_type", data.device_type)
            .tag("node", data.node)
            .field("v", data.v)
            .field("i", data.i)
            .field("p", data.p)
            .field("q", data.q)
            .field("kwh", data.kwh)
            .time(point_time)
        )
        logger.debug("Constructed InfluxDB Point for /fulldatatest: %s", point.to_line_protocol())
        write_api.write(bucket=bucket, record=point)
        logger.info("Successfully wrote full data to InfluxDB for node=%s", data.node)
        return {"status": "ok", "node": data.node}
    except Exception as e:
        logger.error("Failed to write full data to InfluxDB: %s", str(e), exc_info=True)
        return {"status": "error", "error": str(e)}
