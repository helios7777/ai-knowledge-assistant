from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.orchestrator.chains import orchestrator
from app.monitoring.metrics import metrics_db

mcp_router = APIRouter()

class MCPCommand(BaseModel):
    command: str
    args: dict

class MCPResponse(BaseModel):
    status: str
    result: dict

@mcp_router.post("/mcp/execute", response_model=MCPResponse)
async def execute_mcp_command(cmd: MCPCommand):
    
    if cmd.command == "query-docs":
        query = cmd.args.get("query", "")
        result = orchestrator.rag_chain(query)
        
        metrics_db.log_metric(
            tool="rag",
            latency=result["latency"],
            query=query,
            result=result["answer"],
            confidence=result["confidence"]
        )
        
        return MCPResponse(status="success", result=result)
    
    elif cmd.command == "summarize":
        text = cmd.args.get("text", "")
        result = orchestrator.summarize_chain(text)
        
        metrics_db.log_metric(
            tool="summarizer",
            latency=result["latency"],
            query=text[:100],
            result=result["summary"]
        )
        
        return MCPResponse(status="success", result=result)
    
    elif cmd.command == "translate":
        text = cmd.args.get("text", "")
        result = orchestrator.translate_chain(text)
        
        metrics_db.log_metric(
            tool="translator",
            latency=result["latency"],
            query=text[:100],
            result=result["translation"]
        )
        
        return MCPResponse(status="success", result=result)
    
    else:
        return MCPResponse(status="error", result={"message": "Unknown command"})

@mcp_router.get("/mcp/commands")
async def list_commands():
    return {
        "commands": [
            {
                "name": "query-docs",
                "description": "Query the RAG system",
                "args": {"query": "string"}
            },
            {
                "name": "summarize",
                "description": "Summarize text",
                "args": {"text": "string"}
            },
            {
                "name": "translate",
                "description": "Translate English to French",
                "args": {"text": "string"}
            }
        ]
    }