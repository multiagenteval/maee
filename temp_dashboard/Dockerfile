FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and preprocessed data
COPY . .

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"] 