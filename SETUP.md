# RAG AI Voice Assistant - Setup Guide

This guide provides step-by-step instructions to set up and run both the frontend and backend of the RAG AI Voice Assistant.

## Prerequisites

- Python 3.8 or higher
- Node.js 16.x or higher
- npm or yarn
- Git

## Backend Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/Hexinator12/Star.git
   cd Star
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory with the following variables:
   ```
   QDRANT_URL=http://localhost:6333
   # Add other environment variables as needed
   ```

5. **Run the backend server**:
   
   You have two options to run the backend:

   **Option 1: Using Uvicorn directly (Recommended for development)**
   ```bash
   uvicorn api:app --reload
   ```
   - `uvicorn`: ASGI server implementation
   - `api:app`: Tells Uvicorn to look for an ASGI application called `app` in the `api.py` file
   - `--reload`: Enables auto-reload on code changes (development only)
   - **Pros**:
     - Better performance as it's built specifically for ASGI applications
     - Hot reloading for development
     - Better handling of concurrent connections
     - Production-ready with additional configuration

   **Option 2: Using Python directly (Not recommended for production)**
   ```bash
   python api.py
   ```
   - This will only work if you have a `if __name__ == "__main__":` block in your `api.py`
   - **Pros**:
     - Simpler to type
   - **Cons**:
     - Uses Python's built-in development server (not suitable for production)
     - No automatic reload on code changes
     - Slower performance
     - Not optimized for concurrent connections

   The API will be available at `http://localhost:8000`
   - API Docs (Swagger UI): `http://localhost:8000/docs`
   - API Docs (ReDoc): `http://localhost:8000/redoc`

## Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   # or
   yarn dev
   ```
   The frontend will be available at `http://localhost:3000`

## Running the Full Application

1. **Start the backend server** (in a terminal window):
   ```bash
   # From project root
   uvicorn api:app --reload
   ```

2. **Start the frontend development server** (in a separate terminal window):
   ```bash
   # From frontend directory
   cd frontend
   npm run dev
   ```

3. **Access the application**:
   Open your browser and navigate to `http://localhost:3000`

## Development

- **Backend**: The backend will automatically reload when you make changes to the code (thanks to `--reload` flag).
- **Frontend**: The frontend will automatically reload when you make changes to the source files.

## Production Deployment

For production deployment, consider using:
- **Backend**: 
  - Gunicorn with Uvicorn workers for better performance and reliability
  - Example command: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app`
  - This runs 4 worker processes for handling concurrent requests
- **Frontend**: 
  - Build the optimized production bundle with `npm run build`
  - Serve the static files using a production web server
- **Web server**: 
  - Nginx or Apache as a reverse proxy in front of your application
  - Handles SSL termination, static file serving, and load balancing
- **Process manager**: 
  - PM2 or systemd for process management and automatic restarts
  - Ensures your application stays running and restarts if it crashes

## Troubleshooting

1. **Port conflicts**:
   - Backend runs on port 8000 by default
   - Frontend runs on port 3000 by default
   
   Change these in `api.py` and `frontend/vite.config.js` if needed.

2. **Missing dependencies**:
   - Run `pip install -r requirements.txt` for Python dependencies
   - Run `npm install` in the frontend directory for Node.js dependencies

3. **Environment variables**:
   Make sure all required environment variables are set in the `.env` file.
