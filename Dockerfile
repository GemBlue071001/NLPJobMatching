# Use an official Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install dependencies from requirements.txt
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Run the main Python script when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
