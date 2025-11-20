# Use Python 3.10 Slim version
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (including models)
COPY . .

# Expose port 8000
EXPOSE 8000

# Start command
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]