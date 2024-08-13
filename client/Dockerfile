FROM python:3.11.9-slim

# Copy requirements file
COPY requirements.txt .

# Update pip
RUN pip --timeout=3000 install --no-cache-dir --upgrade pip

# Install dependecies
RUN pip --timeout=3000 install --no-cache-dir -r requirements.txt

# Make project directory
RUN mkdir -p /client/

# Set working directory
WORKDIR /client

# Copy client frontend
COPY . .

# # Expose app port Huggingface 
# EXPOSE 7860

# Expose app port
EXPOSE 3939

# Start application
CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "3939"]