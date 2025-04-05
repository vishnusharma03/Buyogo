#!/bin/bash

# Log start of script execution
echo "Starting database setup in Docker container..."

# Run the database.py script
python /app/database.py

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Database setup completed successfully"
else
    echo "Error: Database setup failed"
    exit 1
fi

echo "Script execution completed"

# Use PORT environment variable provided by Render, or default to 9000 if not set
PORT="${PORT:-9000}"
echo "Starting server on port $PORT..."

exec uvicorn app:app --host 0.0.0.0 --port $PORT