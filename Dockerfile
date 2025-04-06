FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install build dependencies for pysqlite3, install requirements, then remove build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc libsqlite3-dev curl \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y --auto-remove gcc libsqlite3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install smolagents[litellm] \
    && mkdir -p /app/Database

COPY data/hotel_bookings.csv Data/
COPY src/ src/


EXPOSE ${PORT}

CMD ["bash", "-c", "cd /app && python /app/src/database.py && uvicorn src.app:app --host 0.0.0.0 --port ${PORT} --reload"]
# CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]