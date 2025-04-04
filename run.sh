#!/bin/bash

# Log start of script execution
echo "Starting database setup in Docker container..."

# Check if data directory exists
# if [ ! -d "/home/vishnusharma7/Buyogo/Data" ]; then
#     echo "Creating Data directory..."
#     mkdir -p /home/vishnusharma7/Buyogo/Data
# fi

# # Check if the CSV file exists
# if [ ! -f "/home/vishnusharma7/Buyogo/Data/hotel_bookings.csv" ]; then
#     echo "Warning: CSV file not found at expected location"
#     # If there's a backup location, you could copy from there
#     if [ -f "/home/vishnusharma7/Buyogo/data/hotel_bookings.csv" ]; then
#         echo "Found CSV in alternate location, copying..."
#         cp /home/vishnusharma7/Buyogo/data/hotel_bookings.csv /home/vishnusharma7/Buyogo/Data/
#     fi
# fi

# Run the database.py script
# echo "Running database.py..."
python /home/vishnusharma7/Buyogo/database.py

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Database setup completed successfully"
else
    echo "Error: Database setup failed"
    exit 1
fi

echo "Script execution completed"