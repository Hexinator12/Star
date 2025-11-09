"""
FastAPI Backend for RAG AI Voice Assistant
Provides REST API endpoints to interact with the assistant from a web frontend.
"""

import sys
import io
# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from AIVoiceAssistant_new import AIVoiceAssistant
import traceback

# Initialize FastAPI app
app = FastAPI(
    title="RAG AI Assistant API",
    description="REST API for University RAG Chatbot",
    version="1.0.0"
)

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global assistant instance
assistant: Optional[AIVoiceAssistant] = None

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "How many programs are there?"
            }
        }

class ChatResponse(BaseModel):
    response: str
    success: bool
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    services: dict
    message: str

class ReloadResponse(BaseModel):
    success: bool
    message: str


# Startup event - Initialize the assistant
@app.on_event("startup")
async def startup_event():
    """Initialize the AI Assistant on server startup."""
    global assistant
    try:
        print("🚀 Initializing AI Voice Assistant...")
        assistant = AIVoiceAssistant("university_dataset_advanced.json")
        print("✅ AI Voice Assistant initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize assistant: {e}")
        traceback.print_exc()
        assistant = None


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG AI Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Send a message to the assistant",
            "GET /health": "Check API and services health",
            "POST /reload": "Reload the knowledge base",
            "GET /docs": "Interactive API documentation"
        }
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health status of the API and required services.
    Returns status of Qdrant, Ollama, and the assistant.
    """
    try:
        if assistant is None:
            return HealthResponse(
                status="unhealthy",
                services={
                    "assistant": "not_initialized",
                    "qdrant": "unknown",
                    "ollama": "unknown"
                },
                message="Assistant not initialized. Check if Qdrant and Ollama are running."
            )
        
        # Check if assistant is properly initialized
        services_status = {
            "assistant": "healthy",
            "qdrant": "healthy" if assistant._client else "unavailable",
            "ollama": "healthy" if assistant.llm else "unavailable"
        }
        
        all_healthy = all(status == "healthy" for status in services_status.values())
        
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            services=services_status,
            message="All services operational" if all_healthy else "Some services unavailable"
        )
        
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            services={"error": str(e)},
            message=f"Health check failed: {str(e)}"
        )


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the AI assistant and get a response.
    
    The assistant uses smart routing:
    - University-specific queries → RAG pipeline (Qdrant + LLM)
    - General conversation → Direct LLM
    """
    try:
        if assistant is None:
            raise HTTPException(
                status_code=503,
                detail="Assistant not initialized. Please check if Qdrant and Ollama are running."
            )
        
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )
        
        # Get response from assistant
        response = assistant.interact_with_llm(request.message.strip())
        
        return ChatResponse(
            response=response,
            success=True,
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        traceback.print_exc()
        return ChatResponse(
            response="I apologize, but I encountered an error processing your request. Please try again.",
            success=False,
            error=str(e)
        )


# Reload knowledge base endpoint
@app.post("/reload", response_model=ReloadResponse)
async def reload_knowledge_base():
    """
    Reload the knowledge base from the JSON file.
    This recreates the Qdrant collection and re-indexes all documents.
    """
    try:
        if assistant is None:
            raise HTTPException(
                status_code=503,
                detail="Assistant not initialized"
            )
        
        print("🔄 Reloading knowledge base...")
        assistant.create_kb()
        print("✅ Knowledge base reloaded successfully!")
        
        return ReloadResponse(
            success=True,
            message="Knowledge base reloaded successfully"
        )
        
    except Exception as e:
        print(f"❌ Error reloading knowledge base: {e}")
        traceback.print_exc()
        return ReloadResponse(
            success=False,
            message=f"Failed to reload knowledge base: {str(e)}"
        )


# Get assistant statistics
@app.get("/stats")
async def get_stats():
    """Get statistics about the assistant's usage."""
    try:
        if assistant is None:
            raise HTTPException(
                status_code=503,
                detail="Assistant not initialized"
            )
        
        return {
            "total_queries": assistant.total_queries,
            "cache_hits": assistant.cache_hits,
            "cache_size": len(assistant.response_cache),
            "cache_hit_rate": f"{(assistant.cache_hits / assistant.total_queries * 100):.2f}%" if assistant.total_queries > 0 else "0%"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


# Run the server
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting RAG AI Assistant API Server")
    print("=" * 60)
    print("\n📋 Prerequisites:")
    print("  1. Qdrant running on http://localhost:6333")
    print("  2. Ollama running with llama2:7b model")
    print("  3. Knowledge base ingested (run ingest_data_to_qdrant.py)")
    print("\n🌐 Server will start on: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
