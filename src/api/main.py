from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.database.schema import (
    EnvironmentalContext,
    StructuredRecommendation,
    ClarificationRequest
)
from src.engine.reasoning import reasoning_engine
from src.engine.memory import ConversationMemory
from src.database.vector_store import vector_store

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-ready REST API for Darukaa.Earth AI Biodiversity Intelligence System."
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory sessions dictionary
sessions: Dict[str, ConversationMemory] = {}

def get_session(session_id: str = "default_session") -> ConversationMemory:
    if session_id not in sessions:
        sessions[session_id] = ConversationMemory()
    return sessions[session_id]

class TextAnalysisRequest(BaseModel):
    query: str = Field(..., description="Natural language query from user")
    session_id: Optional[str] = Field("default_session", description="Session identifier for multi-turn memory")
    additional_context: Optional[EnvironmentalContext] = Field(None, description="Optional structured metric overrides")

class StructuredAnalysisRequest(BaseModel):
    context: EnvironmentalContext = Field(..., description="Structured environmental variables")
    user_query: Optional[str] = Field("Analyze ecosystem degradation and recommend evidence-backed intervention.", description="Specific intent or query")
    session_id: Optional[str] = Field("default_session", description="Session identifier")

class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    top_k: Optional[int] = Field(3, description="Number of scientific snippets to retrieve")

@app.get("/")
def read_root():
    return {
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "vector_store_documents": len(vector_store.documents),
        "docs_url": "/docs"
    }

@app.post("/api/v1/analyze/text", response_model=Dict[str, Any])
def analyze_text(request: TextAnalysisRequest):
    """
    Processes natural language inputs. Extracts metrics, updates session memory,
    checks for missing variables, and returns structured scientific recommendations or clarification requests.
    """
    mem = get_session(request.session_id)
    mem.add_user_message(request.query)
    
    if request.additional_context:
        mem.update_context(request.additional_context)
        
    accumulated_context = mem.get_accumulated_context()
    
    result = reasoning_engine.evaluate(
        user_query=request.query,
        context=accumulated_context,
        chat_history=mem.get_history()
    )
    
    if isinstance(result, StructuredRecommendation):
        mem.add_assistant_message(f"Recommendation: {result.actionable_recommendation}")
        return {
            "type": "recommendation",
            "session_id": request.session_id,
            "accumulated_context": accumulated_context.model_dump(),
            "data": result.model_dump()
        }
    else:
        mem.add_assistant_message(result.clarifying_question)
        return {
            "type": "clarification_request",
            "session_id": request.session_id,
            "accumulated_context": accumulated_context.model_dump(),
            "data": result.model_dump()
        }

@app.post("/api/v1/analyze/structured", response_model=Dict[str, Any])
def analyze_structured(request: StructuredAnalysisRequest):
    """
    Processes structured JSON environmental metric inputs (SOC %, Rainfall, Land Use, Geo-coords).
    Directly evaluates multi-variable scientific interactions.
    """
    mem = get_session(request.session_id)
    mem.update_context(request.context)
    accumulated_context = mem.get_accumulated_context()
    
    result = reasoning_engine.evaluate(
        user_query=request.user_query,
        context=accumulated_context,
        chat_history=mem.get_history()
    )
    
    if isinstance(result, StructuredRecommendation):
        return {
            "type": "recommendation",
            "session_id": request.session_id,
            "accumulated_context": accumulated_context.model_dump(),
            "data": result.model_dump()
        }
    else:
        return {
            "type": "clarification_request",
            "session_id": request.session_id,
            "accumulated_context": accumulated_context.model_dump(),
            "data": result.model_dump()
        }

@app.post("/api/v1/knowledge/search")
def search_knowledge(request: KnowledgeSearchRequest):
    """
    Direct RAG Vector Search endpoint for exploring scientific paper summaries (FAO, IPCC, IPBES).
    """
    results = vector_store.search_similar(request.query, top_k=request.top_k)
    return {
        "query": request.query,
        "count": len(results),
        "results": results
    }

@app.delete("/api/v1/session/{session_id}")
def clear_session(session_id: str):
    """Clears conversation memory and accumulated environmental state."""
    if session_id in sessions:
        sessions[session_id].clear()
        return {"status": "success", "message": f"Session '{session_id}' cleared."}
    return {"status": "not_found", "message": f"Session '{session_id}' does not exist."}
