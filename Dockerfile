FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

EXPOSE 8080

# Specify the command to run on container start
CMD ["streamlit", "run", "Home.py", "--server.port=8080", "--server.address=0.0.0.0"]