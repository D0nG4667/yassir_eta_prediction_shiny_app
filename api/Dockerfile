FROM python:3.11.9-slim

# Copy requirements file
COPY requirements.txt .

# Update pip
RUN pip --timeout=3000 install --no-cache-dir --upgrade pip

# Install dependecies
RUN pip --timeout=3000 install --no-cache-dir -r requirements.txt

# Make project directory
RUN mkdir -p /api/

# Set working directory
WORKDIR /api

# Copy API
COPY . .

# Expose app port
EXPOSE 7860

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]