import requests
from datetime import datetime, timedelta
import time
import random
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIG ---
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID", "YOUR_CHAT_ID")
MESSAGES_PER_DAY = int(os.getenv("MESSAGES_PER_DAY", 5))
START_HOUR = int(os.getenv("START_HOUR", 7))
END_HOUR = int(os.getenv("END_HOUR", 24))

def send_message():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = (f"🔥 Each day is passing...\n"
            f"📘 DSA – where do you stand?\n"
            f"⏰ Time and date: {now}\n"
            f"💼 You will get FAANG, Oracle, or a good product-based company!")
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        response = requests.post(url, data=payload)
        print("Message sent:", response.json())
    except Exception as e:
        print("Error sending message:", e)

def schedule_next_day():
    """Generate MESSAGES_PER_DAY random times between START_HOUR and END_HOUR for today or tomorrow."""
    now = datetime.now()
    start = now.replace(hour=START_HOUR, minute=0, second=0, microsecond=0)
    end = now.replace(hour=END_HOUR, minute=0, second=0, microsecond=0)

    # If we're already past today's END_HOUR, schedule for tomorrow
    if now >= end:
        start += timedelta(days=1)
        end += timedelta(days=1)

    # Generate random times
    times = []
    for _ in range(MESSAGES_PER_DAY):
        delta = random.randint(0, int((end - start).total_seconds()))
        times.append(start + timedelta(seconds=delta))

    times.sort()
    return times

def main():
    scheduled_times = schedule_next_day()
    print("Scheduled times:", [t.strftime("%H:%M:%S") for t in scheduled_times])

    while True:
        now = datetime.now()

        # If all messages for today are sent, reschedule for tomorrow
        if now > scheduled_times[-1]:
            scheduled_times = schedule_next_day()
            print("New schedule:", [t.strftime("%H:%M:%S") for t in scheduled_times])

        # Check if current time matches a scheduled message
        for t in scheduled_times:
            if now >= t and now < t + timedelta(seconds=30):
                send_message()
                # Remove the time once used
                scheduled_times.remove(t)
                break

        time.sleep(20)  # Check every 20 seconds

if __name__ == "__main__":
    main()
