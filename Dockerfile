FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask

# Copy source code
COPY src/ ./src/
COPY app.py .

# Expose port
EXPOSE 5000

# Run the microservice
CMD ["python", "app.py"]
