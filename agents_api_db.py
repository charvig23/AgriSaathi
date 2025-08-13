# agents_api_db.py
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, String, Text, JSON, ForeignKey, Boolean, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from datetime import datetime
import uuid
import faiss
import numpy as np
from enum import Enum
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ----------------------------
# DB Setup
# ----------------------------
# Read DATABASE_URL from .env file
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/agrisathi")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ----------------------------
# ORM Models
# ----------------------------
class Agent(Base):
    __tablename__ = "agents"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    tools = relationship("Tool", back_populates="agent", cascade="all, delete")
    tasks = relationship("Task", back_populates="agent", cascade="all, delete")

class Tool(Base):
    __tablename__ = "tools"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    config = Column(JSON)
    agent_id = Column(String, ForeignKey("agents.id"))
    agent = relationship("Agent", back_populates="tools")

class Task(Base):
    __tablename__ = "tasks"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    status = Column(String, default="pending")
    assigned_tool_id = Column(String, ForeignKey("tools.id"), nullable=True)
    agent_id = Column(String, ForeignKey("agents.id"))
    agent = relationship("Agent", back_populates="tasks")

# agents_api_db.py
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, String, Text, JSON, ForeignKey, Boolean, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from datetime import datetime
import uuid
import faiss
import numpy as np
from enum import Enum

# ----------------------------
# Enums
# ----------------------------
class AgentStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class ToolType(str, Enum):
    API = "api"
    DATABASE = "database"
    LLM = "llm"
    CUSTOM = "custom"

# ----------------------------
# DB Setup
# ----------------------------
# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()

# Read DATABASE_URL from .env file
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/agrisathi")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ----------------------------
# Enhanced ORM Models
# ----------------------------
class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    version = Column(String, default="1.0.0")
    status = Column(String, default=AgentStatus.DRAFT)
    input_schema = Column(JSON)  # JSON schema for agent inputs
    output_schema = Column(JSON)  # JSON schema for agent outputs
    system_prompt = Column(Text)
    instructions = Column(Text)
    capabilities = Column(JSON)  # List of capabilities
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime, nullable=True)
    
    # Relationships
    tools = relationship("Tool", back_populates="agent", cascade="all, delete")
    tasks = relationship("Task", back_populates="agent", cascade="all, delete")

class Tool(Base):
    __tablename__ = "tools"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    tool_type = Column(String, default=ToolType.CUSTOM)
    config = Column(JSON)  # Tool configuration
    input_schema = Column(JSON)  # JSON schema for tool inputs
    output_schema = Column(JSON)  # JSON schema for tool outputs
    implementation = Column(Text)  # Code/implementation details
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Key
    agent_id = Column(String, ForeignKey("agents.id"))
    
    # Relationships
    agent = relationship("Agent", back_populates="tools")

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    status = Column(String, default=TaskStatus.PENDING)
    priority = Column(Integer, default=5)  # 1-10 priority scale
    input_data = Column(JSON)  # Input data for the task
    output_data = Column(JSON)  # Output data from the task
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Foreign Keys
    assigned_tool_id = Column(String, ForeignKey("tools.id"), nullable=True)
    agent_id = Column(String, ForeignKey("agents.id"))
    
    # Relationships
    agent = relationship("Agent", back_populates="tasks")

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
# Enhanced Pydantic Schemas
# ----------------------------
class ToolSchema(BaseModel):
    id: Optional[str] = None
    name: str
    description: str
    tool_type: ToolType = ToolType.CUSTOM
    config: Dict[str, Any] = {}
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    implementation: Optional[str] = None
    is_enabled: bool = True

class ToolResponseSchema(ToolSchema):
    created_at: datetime
    updated_at: datetime
    agent_id: str

class TaskSchema(BaseModel):
    id: Optional[str] = None
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = Field(default=5, ge=1, le=10)
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    assigned_tool_id: Optional[str] = None

class TaskResponseSchema(TaskSchema):
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    agent_id: str

class AgentSchema(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    status: AgentStatus = AgentStatus.DRAFT
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    instructions: Optional[str] = None
    capabilities: Optional[List[str]] = []

class AgentResponseSchema(AgentSchema):
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None
    tools: List[ToolResponseSchema] = []
    tasks: List[TaskResponseSchema] = []

class AgentPublishSchema(BaseModel):
    version: Optional[str] = None
    notes: Optional[str] = None

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(
    title="AgriSaathi Agent Management API", 
    version="2.0.0",
    description="Comprehensive CRUD system for managing AI agents, tools, and tasks"
)

# Dependency: DB Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------------------
# Agent CRUD Endpoints
# ----------------------------
@app.post("/agents", response_model=AgentResponseSchema, status_code=status.HTTP_201_CREATED)
def create_agent(agent: AgentSchema, db: Session = Depends(get_db)):
    """Create a new agent with comprehensive schema validation"""
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
        capabilities=agent.capabilities
    )
    db.add(new_agent)
    db.commit()
    db.refresh(new_agent)
    return new_agent

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
def update_agent(agent_id: str, updated: AgentSchema, db: Session = Depends(get_db)):
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
def create_tool(agent_id: str, tool: ToolSchema, db: Session = Depends(get_db)):
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
def update_tool(agent_id: str, tool_id: str, updated: ToolSchema, db: Session = Depends(get_db)):
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
def create_task(agent_id: str, task: TaskSchema, db: Session = Depends(get_db)):
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
def update_task(agent_id: str, task_id: str, updated: TaskSchema, db: Session = Depends(get_db)):
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
