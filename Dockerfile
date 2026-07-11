FROM python:3.11-slim

# Install Node.js for the frontend build
RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
RUN apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy python dependencies if you have them (e.g. requirements.txt)
# COPY requirements.txt .
# RUN pip install -r requirements.txt
# Since no requirements.txt is provided, we install pandas and numpy directly
RUN pip install pandas numpy scipy matplotlib

# Copy backend and frontend source
COPY . .

# Build frontend
WORKDIR /app/web
RUN npm install
RUN npm run build

# Go back to app root
WORKDIR /app

# Expose port (adjust depending on how the backend will be served, e.g. FastAPI/Flask)
EXPOSE 8000

# Default command
CMD ["python3", "-m", "http.server", "8000"]
