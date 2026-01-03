FROM pathwaycom/pathway:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    poppler-utils \
    libreoffice \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create cache directory with proper permissions
RUN mkdir -p /app/Cache && chmod 777 /app/Cache

CMD ["python", "news_app.py"]