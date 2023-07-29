#!/bin/sh

CONTAINER_ALREADY_STARTED="CONTAINER_ALREADY_STARTED_PLACEHOLDER"
if [ ! -e $CONTAINER_ALREADY_STARTED ]; then
    touch $CONTAINER_ALREADY_STARTED
    echo "-- First startup, initializing database -- This part is still WIP"
    #cat /app/$INIT_DATA | python3 /app/manage.py shell
    
else
    echo "-- Not first container startup, skipping initialization --"
fi
# run the server
echo "-- Starting development server --"
python3 /app/manage.py runserver 0.0.0.0:8000
