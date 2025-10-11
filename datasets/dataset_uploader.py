import os
import time
import logging
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv

load_dotenv()

INFLUXDB_TOKEN = os.environ.get("INFLUXCLIENT") or os.environ.get("INFLUXDB_TOKEN")
INFLUXDB_URL = os.environ.get("INFLUXDB_URL") or "https://influxfinal.atherpulse.in"
INFLUXDB_ORG = "myorg"
BUCKET = "energy4"
CSV_PATH = "/home/amarjith/codespaces/final_yr_project/datasets/training_datasets/residential/residential_load_data_typical_3person_3.5kW_90days_summer.csv"
POINT_NAME = "residentialdataset2"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dataset_uploader")

def main():
    logger.info("Starting dataset_uploader")
    start_time = time.time()

    if not INFLUXDB_TOKEN:
        logger.error("INFLUXDB_TOKEN not found. Aborting.")
        return

    logger.info(f"Using token: {INFLUXDB_TOKEN[:10]}...{INFLUXDB_TOKEN[-10:]}")

    if not os.path.exists(CSV_PATH):
        logger.error("CSV file does not exist: %s", CSV_PATH)
        return

    try:
        logger.debug("Reading CSV from %s", CSV_PATH)
        df = pd.read_csv(CSV_PATH)
        logger.info("CSV loaded: rows=%d, columns=%d", len(df), len(df.columns))
        logger.debug("Columns: %s", list(df.columns))
        
        if 'timestamp' in df.columns:
            logger.debug("Parsing timestamp column")
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            nat_count = int(df['timestamp'].isna().sum())
            logger.info("Timestamps parsed: NaT_count=%d", nat_count)
            
            # **FIX 3: Validate timestamp range**
            ts_min = df['timestamp'].min()
            ts_max = df['timestamp'].max()
            logger.info(f"Data timestamp range: {ts_min} to {ts_max}")
            
            now = pd.Timestamp.now(tz='UTC')
            if ts_max < now - pd.Timedelta(days=365):
                logger.warning(f"⚠️  Data is >1 year old! Latest: {ts_max}")
            if ts_min > now + pd.Timedelta(days=1):
                logger.warning(f"⚠️  Data is in the future! Earliest: {ts_min}")
                
    except Exception as e:
        logger.exception("Failed to read/parse CSV: %s", e)
        return

    client = None
    write_api = None
    
    try:
        logger.debug("Connecting to InfluxDB at %s", INFLUXDB_URL)
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        
        # Health check
        try:
            health = client.health()
            logger.info("InfluxDB health: %s", getattr(health, "status", str(health)))
        except Exception as e:
            logger.warning("Health check failed: %s", e)

        # **FIX 4: Verify bucket access**
        try:
            buckets_api = client.buckets_api()
            all_buckets = buckets_api.find_buckets().buckets
            
            logger.info("=" * 60)
            logger.info("BUCKETS ACCESSIBLE WITH THIS TOKEN:")
            for b in all_buckets:
                logger.info(f"  ✓ {b.name} (org_id: {b.org_id})")
            logger.info("=" * 60)
            
            bucket_names = [b.name for b in all_buckets]
            if BUCKET not in bucket_names:
                logger.error(f"❌ Bucket '{BUCKET}' NOT accessible!")
                logger.error("❌ TOKEN DOES NOT HAVE ACCESS TO THIS BUCKET")
                logger.error("Fix: Regenerate token with write permissions for bucket '%s'", BUCKET)
                return
            else:
                logger.info(f"✓ Bucket '{BUCKET}' is accessible")
                
        except Exception as e:
            logger.exception("Failed to verify bucket access: %s", e)
            return

        # **FIX 2: Use SYNCHRONOUS write for immediate errors**
        write_api = client.write_api(write_options=SYNCHRONOUS)

        total_rows = len(df)
        batch_size = 500
        points = []
        written_points = 0
        batch_count = 0

        for idx, row in df.iterrows():
            point = Point(POINT_NAME)
            
            if 'timestamp' in df.columns and pd.notna(row.get('timestamp')):
                try:
                    ts = row['timestamp']
                    if hasattr(ts, "to_pydatetime"):
                        dt = ts.to_pydatetime()
                    else:
                        dt = pd.to_datetime(ts, utc=True).to_pydatetime()
                    point.time(dt, write_precision=WritePrecision.NS)
                except Exception as e:
                    logger.debug("Timestamp error row %d: %s", idx, e)
            
            for col in df.columns:
                if col == 'timestamp':
                    continue
                try:
                    val = row[col]
                    if pd.isna(val):
                        continue
                    point.field(col, float(val))  # Ensure numeric type
                except Exception as e:
                    logger.debug("Field '%s' error row %d: %s", col, idx, e)
            
            points.append(point)

            if (idx + 1) % 1000 == 0:
                logger.info("Processed %d/%d rows", idx + 1, total_rows)

            if len(points) >= batch_size:
                batch_count += 1
                batch_start = time.time()
                try:
                    # Show sample line protocol
                    if batch_count == 1:
                        sample = points[0].to_line_protocol()
                        logger.debug(f"Sample line protocol:\n{sample}")
                    
                    # Write batch - SYNCHRONOUS will raise exception on failure
                    write_api.write(bucket=BUCKET, org=INFLUXDB_ORG, record=points)
                    written_points += len(points)
                    logger.info(
                        "✓ Wrote batch %d: points=%d (total=%d) in %.3fs",
                        batch_count, len(points), written_points, time.time() - batch_start
                    )
                except Exception as e:
                    logger.exception("❌ WRITE FAILED for batch %d: %s", batch_count, e)
                    logger.error("This typically means:")
                    logger.error("  1. Token lacks WRITE permission")
                    logger.error("  2. Bucket/org mismatch")
                    logger.error("  3. Invalid data format")
                    raise
                
                points = []

        # Write remaining
        if points:
            batch_count += 1
            try:
                write_api.write(bucket=BUCKET, org=INFLUXDB_ORG, record=points)
                written_points += len(points)
                logger.info("✓ Wrote final batch %d: points=%d (total=%d)",
                           batch_count, len(points), written_points)
            except Exception as e:
                logger.exception("❌ WRITE FAILED for final batch: %s", e)
                raise

        # **FIX 2: Explicit flush (for synchronous mode this is less critical)**
        logger.info("Waiting 3 seconds for data to be indexed...")
        time.sleep(3)

        # Better verification query
        try:
            logger.debug("Running verification query")
            query_api = client.query_api()
            
            # Use actual data time range
            start_time_str = df['timestamp'].min().isoformat()
            stop_time_str = df['timestamp'].max().isoformat()
            
            flux = f'''
from(bucket: "{BUCKET}")
  |> range(start: {start_time_str}, stop: {stop_time_str})
  |> filter(fn: (r) => r._measurement == "{POINT_NAME}")
  |> count()
'''
            
            logger.debug(f"Query:\n{flux}")
            tables = query_api.query(flux, org=INFLUXDB_ORG)
            
            if tables and len(tables) > 0:
                total_found = sum(
                    sum(record.get_value() for record in table.records)
                    for table in tables
                )
                logger.info("=" * 60)
                logger.info(f"✓✓✓ VERIFICATION SUCCESS: {total_found} points found!")
                logger.info(f"    Expected: {written_points} points")
                if total_found == written_points * (len(df.columns) - 1):  # Each field creates a point
                    logger.info("    ✓ Count matches (each field creates a point)")
                logger.info("=" * 60)
            else:
                logger.warning("=" * 60)
                logger.warning("❌ NO DATA FOUND IN VERIFICATION QUERY")
                logger.warning("This usually means:")
                logger.warning("  1. Token has WRITE but not READ permission")
                logger.warning("  2. Data written to different org/bucket")
                logger.warning("  3. Timestamp issue (check Data Explorer manually)")
                logger.warning("=" * 60)
                
        except Exception as e:
            logger.exception("Verification query failed: %s", e)

        elapsed = time.time() - start_time
        logger.info("Upload complete: elapsed=%.3fs", elapsed)

    except Exception as e:
        logger.exception("FATAL ERROR: %s", e)
        raise
    
    finally:
        if write_api:
            try:
                logger.debug("Closing write_api")
                write_api.close()
            except:
                pass
                
        if client:
            try:
                logger.debug("Closing client")
                client.close()
            except:
                pass

if __name__ == "__main__":
    main()
