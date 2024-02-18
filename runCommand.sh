#!/bin/bash

# Build the Docker image
docker build -t watch-defect-app .

# Stop and remove the previous container if it exists
docker ps | grep "watch-defect-app" && docker stop watch-defect-app
docker ps -a | grep "watch-defect-app" && docker rm watch-defect-app

# Run the Docker container with GPU support and restart policy
docker run -d --gpus all --ipc=host -p 80:8080 --restart=always --name watch-defect-app watch-defect-app

# Follow the logs with a timestamp since the last minute
docker logs -f --since 1m watch-defect-app
