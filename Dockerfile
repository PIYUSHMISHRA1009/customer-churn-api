# 1. Base image: Start from official Python 3.9 slim image (small footprint)
FROM python:3.9-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file first for dependency installation
COPY requirements-docker.txt .

# 4. Upgrade pip (optional, but good practice)
RUN pip install --upgrade pip

# 5. Install Python dependencies from requirements file
RUN pip install -r requirements-docker.txt

# 6. Copy entire project contents into the container’s /app directory
COPY . .

# 7. Expose port 8000 so Docker knows the container listens on this port
EXPOSE 8000

# 8. Command to run when container starts:
#    Starts uvicorn server hosting your FastAPI app on all interfaces (0.0.0.0)
CMD ["uvicorn", "src.inference_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
