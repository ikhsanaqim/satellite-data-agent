# Gunakan base image Python yang ringan
FROM python:3.11-slim

# Konfigurasi environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies yang mungkin dibutuhkan (terutama untuk C++ extensions seperti ChromaDB/pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tambahkan user 1000 untuk Hugging Face (wajib agar ChromaDB bisa menulis ke disk)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy file requirements terlebih dahulu untuk memanfaatkan Docker cache
COPY --chown=user requirements.txt .

# Install library Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code project ke dalam container
COPY --chown=user . $HOME/app

# Install project ini sendiri sebagai local package (agar src/ bisa di-import)
RUN pip install --no-cache-dir -e .

# Expose port 7860 untuk Hugging Face
EXPOSE 7860

# Command untuk menjalankan Streamlit saat container dijalankan
CMD streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false
