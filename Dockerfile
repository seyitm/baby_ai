# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Expose port 8080 to let Railway know which port to listen to
EXPOSE 8080

# Command to run the application
# Railway provides the PORT environment variable, but we default to 8080 for consistency.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
