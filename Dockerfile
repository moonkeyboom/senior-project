FROM python:3.11-slim



# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and frontend source
COPY . .



# Expose port (adjust depending on how the backend will be served, e.g. FastAPI/Flask)
EXPOSE 8000

# Default command to run FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
