# FastAPI Backend Guide

## 🚀 Quick Start

### Prerequisites
1. **Qdrant** running on `http://localhost:6333`
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Ollama** with llama2:7b model
   ```bash
   ollama pull llama2:7b
   ```

3. **Knowledge Base** ingested to Qdrant
   ```bash
   python ingest_data_to_qdrant.py --kb-path voice_rag_kb.json --force
   ```

### Installation

Install the new dependencies:
```bash
pip install fastapi uvicorn[standard] pydantic
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### Running the API Server

```bash
python api.py
```

The server will start on: **http://localhost:8000**

---

## 📚 API Endpoints

### 1. **Root** - `GET /`
Get API information and available endpoints.

**Example:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "RAG AI Assistant API",
  "version": "1.0.0",
  "endpoints": {
    "POST /chat": "Send a message to the assistant",
    "GET /health": "Check API and services health",
    "POST /reload": "Reload the knowledge base",
    "GET /docs": "Interactive API documentation"
  }
}
```

---

### 2. **Health Check** - `GET /health`
Check if all services (Qdrant, Ollama, Assistant) are running properly.

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "assistant": "healthy",
    "qdrant": "healthy",
    "ollama": "healthy"
  },
  "message": "All services operational"
}
```

---

### 3. **Chat** - `POST /chat`
Send a message to the assistant and get a response.

**Request Body:**
```json
{
  "message": "How many programs are there?"
}
```

**Example with curl:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many programs are there?"}'
```

**Example with PowerShell:**
```powershell
$body = @{
    message = "How many programs are there?"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/chat" -Method Post -Body $body -ContentType "application/json"
```

**Response:**
```json
{
  "response": "There are 6 academic programs in total.",
  "success": true,
  "error": null
}
```

---

### 4. **Reload Knowledge Base** - `POST /reload`
Reload the knowledge base from the JSON file.

**Example:**
```bash
curl -X POST http://localhost:8000/reload
```

**Response:**
```json
{
  "success": true,
  "message": "Knowledge base reloaded successfully"
}
```

---

### 5. **Statistics** - `GET /stats`
Get usage statistics about the assistant.

**Example:**
```bash
curl http://localhost:8000/stats
```

**Response:**
```json
{
  "total_queries": 42,
  "cache_hits": 15,
  "cache_size": 20,
  "cache_hit_rate": "35.71%"
}
```

---

## 🧪 Testing the API

### Using the Interactive Docs

Visit **http://localhost:8000/docs** for Swagger UI with interactive API testing.

### Test Scenarios

1. **Greeting Test:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello"}'
   ```

2. **University Query Test:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What programs do you offer?"}'
   ```

3. **Count Query Test:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "How many courses are there?"}'
   ```

4. **General Question Test:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What is the weather like?"}'
   ```

---

## 🔧 Configuration

### CORS Settings
By default, the API allows requests from any origin (`allow_origins=["*"]`).

For production, update in `api.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Port Configuration
To change the port, modify in `api.py`:
```python
uvicorn.run(
    "api:app",
    host="0.0.0.0",
    port=8000,  # Change this
    reload=True
)
```

---

## 🐛 Troubleshooting

### Error: "Assistant not initialized"
**Cause:** Qdrant or Ollama not running, or knowledge base not ingested.

**Solution:**
1. Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
2. Start Ollama: `ollama serve`
3. Ingest data: `python ingest_data_to_qdrant.py --force`

### Error: "Connection refused"
**Cause:** Qdrant not accessible on port 6333.

**Solution:** Check if Qdrant is running: `curl http://localhost:6333/`

### Error: "Model not found"
**Cause:** Llama2 model not downloaded in Ollama.

**Solution:** `ollama pull llama2:7b`

---

## 📊 Performance

- **Average Response Time:** 2-3 seconds (RAG queries)
- **Cache Hit Rate:** ~30-40% for repeated queries
- **Concurrent Requests:** Supports multiple simultaneous users

---

## 🔐 Security Notes

For production deployment:
1. Add authentication (JWT tokens, API keys)
2. Restrict CORS to specific origins
3. Add rate limiting
4. Use HTTPS
5. Validate and sanitize all inputs
6. Add logging and monitoring

---

## 📝 Next Steps

Once the API is working:
1. Test all endpoints
2. Create the React frontend
3. Connect frontend to this API
4. Deploy both frontend and backend

---

## 🤝 API Response Format

All endpoints follow a consistent response format:

**Success Response:**
```json
{
  "response": "string",
  "success": true,
  "error": null
}
```

**Error Response:**
```json
{
  "response": "Error message",
  "success": false,
  "error": "Detailed error description"
}
```
