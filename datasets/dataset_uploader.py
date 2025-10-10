import os
import time
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WriteOptions
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

INFLUXDB_URL = "https://influxfinal.atherpulse.in"
INFLUXDB_TOKEN = os.environ.get("INFLUXCLIENT")
INFLUXDB_ORG = "myorg"
BUCKET = "energy3"
CSV_PATH = "./serious_datasets/industrial/industrial_load_data_90days.csv"  # Change to your CSV file path
POINT_NAME = "industrialdataset"  # Change to your desired measurement name

def main():
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    write_api = client.write_api(write_options=WriteOptions(batch_size=500, flush_interval=0))

    df = pd.read_csv(CSV_PATH)

    try:
        batch_size = 500
        points = []
        for idx, row in df.iterrows():
            point = Point(POINT_NAME)
            # If 'timestamp' column exists, use it
            if 'timestamp' in row:
                point.time(row['timestamp'])
            # Add all other columns as fields
            for col in df.columns:
                if col != 'timestamp':
                    point.field(col, row[col])
            points.append(point)
            if len(points) == batch_size:
                write_api.write(bucket=BUCKET, org=INFLUXDB_ORG, record=points)
                points = []
        if points:
            write_api.write(bucket=BUCKET, org=INFLUXDB_ORG, record=points)
        write_api.flush()  # Ensure all writes are completed before shutdown
        time.sleep(1)  # Give background threads time to finish
    finally:
        client.close()

if __name__ == "__main__":
    main()