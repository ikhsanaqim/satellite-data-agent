# Gunakan base image Python yang ringan
FROM python:3.11-slim

# Konfigurasi environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory di dalam container
WORKDIR /app

# Install system dependencies yang mungkin dibutuhkan (terutama untuk C++ extensions seperti ChromaDB/pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements terlebih dahulu untuk memanfaatkan Docker cache
COPY requirements.txt .

# Install library Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code project ke dalam container
COPY . .

# Install project ini sendiri sebagai local package (agar src/ bisa di-import)
RUN pip install --no-cache-dir -e .

# Expose port default Streamlit
EXPOSE 8501

# Command untuk menjalankan Streamlit saat container dijalankan
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
