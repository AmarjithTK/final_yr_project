# 1. Start from an official Python base image
# python:3.9-slim is a good choice for being lightweight
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# 2. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the FastAPI application code into the container
COPY main.py .

# 4. Expose the port the app runs on
EXPOSE 8000

# 5. Specify the command to run the application
# Uvicorn is the ASGI server that will run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]