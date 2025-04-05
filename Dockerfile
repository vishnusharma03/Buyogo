FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install build dependencies for pysqlite3, install requirements, then remove build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc libsqlite3-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y --auto-remove gcc libsqlite3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install smolagents[litellm]

COPY data/hotel_bookings.csv Data/
# COPY Database/hotel_database.db Database/
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]