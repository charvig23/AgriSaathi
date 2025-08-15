# agents_api_db.py
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime
import uuid
import faiss
import numpy as np
from enum import Enum
import os
from dotenv import load_dotenv

import openai
# from anthropic import Anthropic  # Commented out - paid service
import google.generativeai as genai
import requests
import json
# Import from our modular files
from database import engine, get_database as get_db, Base
from models import Agent, Tool, Task
from schemas import (
    AgentCreateSchema, AgentUpdateSchema, AgentResponseSchema, AgentPublishSchema,
    ToolCreateSchema, ToolUpdateSchema, ToolResponseSchema,
    TaskCreateSchema, TaskUpdateSchema, TaskResponseSchema,
    ToolSelectionRequest, ToolSelectionResponse,
    AgentStatus, TaskStatus, ToolType, LLMModel,
    SystemStatsSchema, AgentStatsSchema, ToolStatsSchema, TaskStatsSchema
)

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Initialize API clients
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Anthropic is commented out - paid service
# if ANTHROPIC_API_KEY:
#     anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


# Add this LLM integration class
class LLMService:
    @staticmethod
    def call_openai_gpt(model: str, prompt: str, config: dict) -> str:
        """Call OpenAI GPT models"""
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 1024),
                top_p=config.get("top_p", 1.0),
                frequency_penalty=config.get("frequency_penalty", 0),
                presence_penalty=config.get("presence_penalty", 0)
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    
    # Anthropic Claude support removed - paid service
    # @staticmethod
    # def call_anthropic_claude(model: str, prompt: str, config: dict) -> str:
    #     """Call Anthropic Claude models"""
    #     raise HTTPException(status_code=501, detail="Anthropic Claude models not supported - paid service")
    
    @staticmethod
    def call_google_gemini(model: str, prompt: str, config: dict) -> str:
        """Call Google Gemini models"""
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=500, detail="Google API key not configured")
        
        try:
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.get("temperature", 0.7),
                    max_output_tokens=config.get("max_tokens", 1024),
                    top_p=config.get("top_p", 1.0)
                )
            )
            return response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Google API error: {str(e)}")
    
    @staticmethod
    def call_local_model(model: str, prompt: str, config: dict) -> str:
        """Call local models via Ollama"""
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": config.get("temperature", 0.7),
                        "num_predict": config.get("max_tokens", 1024),
                        "top_p": config.get("top_p", 1.0)
                    }
                }
            )
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise HTTPException(status_code=500, detail=f"Local model error: {response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Local model error: {str(e)}")

# Update the simple_tool_selection function to use LLM
def llm_based_tool_selection(agent: Agent, user_query: str, tools: List[Tool]) -> dict:
    """
    LLM-based tool selection using the agent's configured model
    """
    if not agent.llm_model or agent.llm_model == LLMModel.NONE:
        return simple_tool_selection(user_query, tools)
    
    # Prepare context for LLM
    tools_context = ""
    for i, tool in enumerate(tools):
        tools_context += f"{i+1}. {tool.name}: {tool.description} (Type: {tool.tool_type})\n"
    
    # Use agent's custom prompt or default
    selection_prompt = agent.tool_selection_prompt or """
You are an intelligent tool selector. Based on the user query and available tools, select the most appropriate tool.

User Query: {user_query}

Available Tools:
{tools_context}

Respond with ONLY the tool number (1, 2, 3, etc.) and a brief reason.
Format: "Tool X: [reason]"
"""
    
    prompt = selection_prompt.format(
        user_query=user_query,
        tools_context=tools_context
    )
    
    try:
        # Route to appropriate LLM based on model type
        model = agent.llm_model.value
        config = agent.llm_config or {}
        
        if model.startswith("gpt"):
            response = LLMService.call_openai_gpt(model, prompt, config)
        elif model.startswith("gemini"):
            response = LLMService.call_google_gemini(model, prompt, config)
        elif model.startswith("claude"):
            # Claude models are not supported (paid service)
            raise HTTPException(status_code=501, detail=f"Claude models ({model}) are not supported - paid service. Please use GPT, Gemini, or local models instead.")
        else:
            # Local models
            response = LLMService.call_local_model(model, prompt, config)
        
        # Parse LLM response to extract tool selection
        import re
        tool_match = re.search(r'Tool (\d+)', response)
        if tool_match:
            tool_index = int(tool_match.group(1)) - 1
            if 0 <= tool_index < len(tools):
                selected_tool = tools[tool_index]
                return {
                    "selected_tool_id": selected_tool.id,
                    "selected_tool_name": selected_tool.name,
                    "confidence_score": 0.9,  # High confidence for LLM selection
                    "reasoning": f"LLM-selected tool based on: {response.strip()}",
                    "fallback_tools": [tool.id for tool in tools if tool.id != selected_tool.id][:2]
                }
        
        # Fallback to simple selection if LLM response parsing fails
        return simple_tool_selection(user_query, tools)
        
    except Exception as e:
        print(f"LLM tool selection failed: {str(e)}")
        # Fallback to simple selection
        return simple_tool_selection(user_query, tools)


# Create database tables
Base.metadata.create_all(bind=engine)

# ----------------------------
# FAISS Index (vector store for RAG)
# ----------------------------
embedding_dim = 384  # Depends on your embedding model
faiss_index = faiss.IndexFlatL2(embedding_dim)
doc_embeddings = {}  # Map doc_id → vector

def add_to_faiss(doc_id: str, vector: np.ndarray):
    faiss_index.add(np.array([vector], dtype=np.float32))
    doc_embeddings[doc_id] = vector

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(
    title="AgriSaathi Agent Management API", 
    version="2.0.0",
    description="Comprehensive CRUD system for managing AI agents, tools, and tasks"
)

# ----------------------------
# Agent CRUD Endpoints
# ----------------------------
@app.post("/agents", response_model=AgentResponseSchema, status_code=status.HTTP_201_CREATED)
def create_agent(agent: AgentCreateSchema, db: Session = Depends(get_db)):
    """Create a new agent with comprehensive schema validation"""
    try:
        new_agent = Agent(
            id=str(uuid.uuid4()),
            name=agent.name,
            description=agent.description,
            version=agent.version,
            status=agent.status,
            input_schema=agent.input_schema,
            output_schema=agent.output_schema,
            system_prompt=agent.system_prompt,
            instructions=agent.instructions,
            capabilities=agent.capabilities,
            # LLM Configuration
            llm_model=agent.llm_model,
            llm_config=agent.llm_config,
            tool_selection_prompt=agent.tool_selection_prompt,
            enable_intelligent_routing=agent.enable_intelligent_routing
        )
        db.add(new_agent)
        db.commit()
        db.refresh(new_agent)
        return new_agent
    except Exception as e:
        db.rollback()
        print(f"Error creating agent: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")

@app.get("/agents", response_model=List[AgentResponseSchema])
def list_agents(
    status_filter: Optional[AgentStatus] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all agents with optional filtering and pagination"""
    query = db.query(Agent)
    if status_filter:
        query = query.filter(Agent.status == status_filter)
    return query.offset(skip).limit(limit).all()

@app.get("/agents/{agent_id}", response_model=AgentResponseSchema)
def get_agent(agent_id: str, db: Session = Depends(get_db)):
    """Get a specific agent by ID"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.put("/agents/{agent_id}", response_model=AgentResponseSchema)
def update_agent(agent_id: str, updated: AgentUpdateSchema, db: Session = Depends(get_db)):
    """Update an existing agent"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Update fields
    for field, value in updated.dict(exclude_unset=True).items():
        if field != "id":  # Don't update ID
            setattr(agent, field, value)
    
    agent.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(agent)
    return agent

@app.post("/agents/{agent_id}/publish", response_model=AgentResponseSchema)
def publish_agent(agent_id: str, publish_data: AgentPublishSchema, db: Session = Depends(get_db)):
    """Publish an agent (change status to published)"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if agent.status == AgentStatus.PUBLISHED:
        raise HTTPException(status_code=400, detail="Agent is already published")
    
    # Update version if provided
    if publish_data.version:
        agent.version = publish_data.version
    
    agent.status = AgentStatus.PUBLISHED
    agent.published_at = datetime.utcnow()
    agent.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(agent)
    return agent

@app.post("/agents/{agent_id}/unpublish", response_model=AgentResponseSchema)
def unpublish_agent(agent_id: str, db: Session = Depends(get_db)):
    """Unpublish an agent (change status to draft)"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent.status = AgentStatus.DRAFT
    agent.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(agent)
    return agent

@app.delete("/agents/{agent_id}")
def delete_agent(agent_id: str, db: Session = Depends(get_db)):
    """Delete an agent and all associated tools and tasks"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    db.delete(agent)
    db.commit()
    return {"message": f"Agent {agent_id} deleted successfully"}

# ----------------------------
# Tool CRUD Endpoints
# ----------------------------
@app.post("/agents/{agent_id}/tools", response_model=ToolResponseSchema, status_code=status.HTTP_201_CREATED)
def create_tool(agent_id: str, tool: ToolCreateSchema, db: Session = Depends(get_db)):
    """Create a new tool for an agent"""
    # Check if agent exists
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    new_tool = Tool(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        name=tool.name,
        description=tool.description,
        tool_type=tool.tool_type,
        config=tool.config,
        input_schema=tool.input_schema,
        output_schema=tool.output_schema,
        implementation=tool.implementation,
        is_enabled=tool.is_enabled
    )
    
    db.add(new_tool)
    db.commit()
    db.refresh(new_tool)
    return new_tool

@app.get("/agents/{agent_id}/tools", response_model=List[ToolResponseSchema])
def get_agent_tools(agent_id: str, enabled_only: bool = False, db: Session = Depends(get_db)):
    """Get all tools for a specific agent"""
    query = db.query(Tool).filter(Tool.agent_id == agent_id)
    if enabled_only:
        query = query.filter(Tool.is_enabled == True)
    return query.all()

@app.get("/tools/{tool_id}", response_model=ToolResponseSchema)
def get_tool(tool_id: str, db: Session = Depends(get_db)):
    """Get a specific tool by ID"""
    tool = db.query(Tool).filter(Tool.id == tool_id).first()
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool

@app.put("/agents/{agent_id}/tools/{tool_id}", response_model=ToolResponseSchema)
def update_tool(agent_id: str, tool_id: str, updated: ToolUpdateSchema, db: Session = Depends(get_db)):
    """Update a specific tool"""
    tool_obj = db.query(Tool).filter(Tool.id == tool_id, Tool.agent_id == agent_id).first()
    if not tool_obj:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    # Update fields
    for field, value in updated.dict(exclude_unset=True).items():
        if field != "id":  # Don't update ID
            setattr(tool_obj, field, value)
    
    tool_obj.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(tool_obj)
    return tool_obj

@app.delete("/agents/{agent_id}/tools/{tool_id}")
def delete_tool(agent_id: str, tool_id: str, db: Session = Depends(get_db)):
    """Delete a specific tool"""
    tool_obj = db.query(Tool).filter(Tool.id == tool_id, Tool.agent_id == agent_id).first()
    if not tool_obj:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    db.delete(tool_obj)
    db.commit()
    return {"message": f"Tool {tool_id} deleted successfully"}

# ----------------------------
# Task CRUD Endpoints
# ----------------------------
@app.post("/agents/{agent_id}/tasks", response_model=TaskResponseSchema, status_code=status.HTTP_201_CREATED)
def create_task(agent_id: str, task: TaskCreateSchema, db: Session = Depends(get_db)):
    """Create a new task for an agent"""
    # Check if agent exists
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Check if assigned tool exists and belongs to the agent
    if task.assigned_tool_id:
        tool = db.query(Tool).filter(
            Tool.id == task.assigned_tool_id,
            Tool.agent_id == agent_id
        ).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Assigned tool not found or doesn't belong to this agent")
    
    new_task = Task(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        name=task.name,
        description=task.description,
        status=task.status,
        priority=task.priority,
        input_data=task.input_data,
        output_data=task.output_data,
        error_message=task.error_message,
        assigned_tool_id=task.assigned_tool_id
    )
    
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    return new_task

@app.get("/agents/{agent_id}/tasks", response_model=List[TaskResponseSchema])
def get_agent_tasks(
    agent_id: str,
    status_filter: Optional[TaskStatus] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all tasks for a specific agent"""
    query = db.query(Task).filter(Task.agent_id == agent_id)
    if status_filter:
        query = query.filter(Task.status == status_filter)
    return query.offset(skip).limit(limit).all()

@app.get("/tasks/{task_id}", response_model=TaskResponseSchema)
def get_task(task_id: str, db: Session = Depends(get_db)):
    """Get a specific task by ID"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.put("/agents/{agent_id}/tasks/{task_id}", response_model=TaskResponseSchema)
def update_task(agent_id: str, task_id: str, updated: TaskUpdateSchema, db: Session = Depends(get_db)):
    """Update a specific task"""
    task_obj = db.query(Task).filter(Task.id == task_id, Task.agent_id == agent_id).first()
    if not task_obj:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check if assigned tool exists and belongs to the agent
    if updated.assigned_tool_id:
        tool = db.query(Tool).filter(
            Tool.id == updated.assigned_tool_id,
            Tool.agent_id == agent_id
        ).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Assigned tool not found or doesn't belong to this agent")
    
    # Update fields
    for field, value in updated.dict(exclude_unset=True).items():
        if field != "id":  # Don't update ID
            setattr(task_obj, field, value)
    
    # Set completion time if task is completed
    if updated.status == TaskStatus.COMPLETED and task_obj.completed_at is None:
        task_obj.completed_at = datetime.utcnow()
    
    task_obj.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(task_obj)
    return task_obj

@app.delete("/agents/{agent_id}/tasks/{task_id}")
def delete_task(agent_id: str, task_id: str, db: Session = Depends(get_db)):
    """Delete a specific task"""
    task_obj = db.query(Task).filter(Task.id == task_id, Task.agent_id == agent_id).first()
    if not task_obj:
        raise HTTPException(status_code=404, detail="Task not found")
    
    db.delete(task_obj)
    db.commit()
    return {"message": f"Task {task_id} deleted successfully"}

# ----------------------------
# Bulk Operations
# ----------------------------
@app.get("/agents/{agent_id}/export", response_model=AgentResponseSchema)
def export_agent(agent_id: str, db: Session = Depends(get_db)):
    """Export agent with all tools and tasks"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.post("/agents/import", response_model=AgentResponseSchema)
def import_agent(agent_data: AgentResponseSchema, db: Session = Depends(get_db)):
    """Import agent with all tools and tasks"""
    # Create new agent
    new_agent = Agent(
        id=str(uuid.uuid4()),  # Generate new ID
        name=agent_data.name,
        description=agent_data.description,
        version=agent_data.version,
        status=AgentStatus.DRAFT,  # Always import as draft
        input_schema=agent_data.input_schema,
        output_schema=agent_data.output_schema,
        system_prompt=agent_data.system_prompt,
        instructions=agent_data.instructions,
        capabilities=agent_data.capabilities
    )
    
    db.add(new_agent)
    db.flush()  # Get the agent ID
    
    # Import tools
    for tool_data in agent_data.tools:
        new_tool = Tool(
            id=str(uuid.uuid4()),
            agent_id=new_agent.id,
            name=tool_data.name,
            description=tool_data.description,
            tool_type=tool_data.tool_type,
            config=tool_data.config,
            input_schema=tool_data.input_schema,
            output_schema=tool_data.output_schema,
            implementation=tool_data.implementation,
            is_enabled=tool_data.is_enabled
        )
        db.add(new_tool)
    
    # Import tasks
    for task_data in agent_data.tasks:
        new_task = Task(
            id=str(uuid.uuid4()),
            agent_id=new_agent.id,
            name=task_data.name,
            description=task_data.description,
            status=TaskStatus.PENDING,  # Reset status
            priority=task_data.priority,
            input_data=task_data.input_data
        )
        db.add(new_task)
    
    db.commit()
    db.refresh(new_agent)
    return new_agent

# ----------------------------
# Health and Statistics
# ----------------------------
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "2.0.0"
    }

@app.get("/stats")
def get_statistics(db: Session = Depends(get_db)):
    """Get system statistics"""
    agent_count = db.query(Agent).count()
    tool_count = db.query(Tool).count()
    task_count = db.query(Task).count()
    published_agents = db.query(Agent).filter(Agent.status == AgentStatus.PUBLISHED).count()
    
    return {
        "agents": {
            "total": agent_count,
            "published": published_agents,
            "draft": agent_count - published_agents
        },
        "tools": {
            "total": tool_count
        },
        "tasks": {
            "total": task_count,
            "pending": db.query(Task).filter(Task.status == TaskStatus.PENDING).count(),
            "completed": db.query(Task).filter(Task.status == TaskStatus.COMPLETED).count(),
            "failed": db.query(Task).filter(Task.status == TaskStatus.FAILED).count()
        }
    }

# ----------------------------
# LLM-based Tool Selection
# ----------------------------
def simple_tool_selection(user_query: str, tools: List[Tool]) -> dict:
    """
    Simple tool selection based on keyword matching.
    In a production environment, this would use an actual LLM.
    """
    if not tools:
        return None
    
    query_lower = user_query.lower()
    
    # Simple scoring based on keywords in tool names and descriptions
    tool_scores = []
    for tool in tools:
        score = 0.0
        
        # Check tool name
        if tool.name.lower() in query_lower:
            score += 0.5
        
        # Check tool description
        if tool.description and any(word in tool.description.lower() for word in query_lower.split()):
            score += 0.3
        
        # Check tool type relevance
        tool_type_keywords = {
            'api': ['api', 'request', 'call', 'fetch', 'get'],
            'database': ['data', 'database', 'query', 'search', 'store'],
            'llm': ['generate', 'text', 'chat', 'language', 'ai'],
            'custom': ['custom', 'specific', 'unique']
        }
        
        for keyword in tool_type_keywords.get(tool.tool_type.value, []):
            if keyword in query_lower:
                score += 0.2
        
        tool_scores.append((tool, score))
    
    # Sort by score and return the best match
    tool_scores.sort(key=lambda x: x[1], reverse=True)
    best_tool, best_score = tool_scores[0]
    
    # Get fallback tools (second and third best)
    fallback_tools = [tool.id for tool, _ in tool_scores[1:3]]
    
    return {
        "selected_tool_id": best_tool.id,
        "selected_tool_name": best_tool.name,
        "confidence_score": min(best_score, 1.0),  # Cap at 1.0
        "reasoning": f"Selected '{best_tool.name}' based on keyword matching and tool type relevance. Score: {best_score:.2f}",
        "fallback_tools": fallback_tools
    }

@app.get("/agents/{agent_id}/llm-config")
def get_agent_llm_config(agent_id: str, db: Session = Depends(get_db)):
    """Get the LLM configuration for an agent"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "agent_id": agent_id,
        "llm_model": agent.llm_model,
        "llm_config": agent.llm_config or {},
        "tool_selection_prompt": agent.tool_selection_prompt,
        "enable_intelligent_routing": agent.enable_intelligent_routing,
        "available_models": [model.value for model in LLMModel]
    }

@app.put("/agents/{agent_id}/llm-config")
def update_agent_llm_config(
    agent_id: str, 
    llm_model: LLMModel,
    llm_config: Optional[Dict[str, Any]] = None,
    tool_selection_prompt: Optional[str] = None,
    enable_intelligent_routing: bool = False,
    db: Session = Depends(get_db)
):
    """Update the LLM configuration for an agent"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent.llm_model = llm_model
    agent.llm_config = llm_config or {}
    agent.enable_intelligent_routing = enable_intelligent_routing
    
    if tool_selection_prompt:
        agent.tool_selection_prompt = tool_selection_prompt
    
    agent.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(agent)
    
    return {
        "message": "LLM configuration updated successfully",
        "agent_id": agent_id,
        "llm_model": agent.llm_model,
        "enable_intelligent_routing": agent.enable_intelligent_routing
    }

# ----------------------------
# Root
# ----------------------------
@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "AgriSaathi Agent Management API",
        "version": "2.0.0",
        "description": "Comprehensive CRUD system for managing AI agents, tools, and tasks",
        "endpoints": {
            "agents": "/agents",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        }
    }

# ----------------------------
# Tool Selection Endpoint (LLM-based)
# ----------------------------
@app.post("/agents/{agent_id}/select-tool", response_model=ToolSelectionResponse)
def select_tool_for_query(
    agent_id: str, 
    request: ToolSelectionRequest, 
    db: Session = Depends(get_db)
):
    """
    Use the agent's LLM model to intelligently select the best tool for a user query.
    Falls back to simple heuristics if LLM is not configured.
    """
    # Get the agent
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Check if intelligent routing is enabled
    if not agent.enable_intelligent_routing:
        raise HTTPException(
            status_code=400, 
            detail="Intelligent tool routing is not enabled for this agent. Enable it in the agent configuration."
        )
    
    # Get agent's tools
    tools = db.query(Tool).filter(Tool.agent_id == agent_id, Tool.is_enabled == True).all()
    if not tools:
        raise HTTPException(status_code=404, detail="No enabled tools found for this agent")
    
    # Use LLM-based tool selection
    selection_result = llm_based_tool_selection(agent, request.user_query, tools)
    
    if not selection_result:
        raise HTTPException(status_code=404, detail="No suitable tool found for the query")
    
    return ToolSelectionResponse(**selection_result)

# ----------------------------
# Run the application
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting AgriSaathi Agent Management API v2.0.0")
    print("API Documentation will be available at: http://localhost:8000/docs")
    print("Alternative docs at: http://localhost:8000/redoc")
    print("-" * 50)
    
    uvicorn.run(
        "agents_api_db:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
