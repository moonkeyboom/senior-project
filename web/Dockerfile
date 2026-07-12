FROM node:20-slim

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source
COPY . .

# Expose Vite's default port
EXPOSE 5173

# Start Vite dev server exposing to network
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
