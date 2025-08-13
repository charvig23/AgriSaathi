"""
Pydantic models for request/response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Enums
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

# Base schemas
class BaseSchema(BaseModel):
    class Config:
        from_attributes = True  # For SQLAlchemy compatibility

# Tool schemas
class ToolCreateSchema(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1)
    tool_type: ToolType = ToolType.CUSTOM
    config: Dict[str, Any] = Field(default_factory=dict)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    implementation: Optional[str] = None
    is_enabled: bool = True

class ToolUpdateSchema(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, min_length=1)
    tool_type: Optional[ToolType] = None
    config: Optional[Dict[str, Any]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    implementation: Optional[str] = None
    is_enabled: Optional[bool] = None

class ToolResponseSchema(BaseSchema):
    id: str
    name: str
    description: str
    tool_type: ToolType
    config: Dict[str, Any]
    input_schema: Optional[Dict[str, Any]]
    output_schema: Optional[Dict[str, Any]]
    implementation: Optional[str]
    is_enabled: bool
    created_at: datetime
    updated_at: datetime
    agent_id: str

# Task schemas
class TaskCreateSchema(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = Field(default=5, ge=1, le=10)
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    assigned_tool_id: Optional[str] = None

class TaskUpdateSchema(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, min_length=1)
    status: Optional[TaskStatus] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    assigned_tool_id: Optional[str] = None

class TaskResponseSchema(BaseSchema):
    id: str
    name: str
    description: str
    status: TaskStatus
    priority: int
    input_data: Optional[Dict[str, Any]]
    output_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    assigned_tool_id: Optional[str]
    agent_id: str

# Agent schemas
class AgentCreateSchema(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    status: AgentStatus = AgentStatus.DRAFT
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    instructions: Optional[str] = None
    capabilities: Optional[List[str]] = Field(default_factory=list)

class AgentUpdateSchema(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    version: Optional[str] = Field(None, pattern=r"^\d+\.\d+\.\d+$")
    status: Optional[AgentStatus] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    instructions: Optional[str] = None
    capabilities: Optional[List[str]] = None

class AgentResponseSchema(BaseSchema):
    id: str
    name: str
    description: Optional[str]
    version: str
    status: AgentStatus
    input_schema: Optional[Dict[str, Any]]
    output_schema: Optional[Dict[str, Any]]
    system_prompt: Optional[str]
    instructions: Optional[str]
    capabilities: Optional[List[str]]
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime]
    tools: List[ToolResponseSchema] = Field(default_factory=list)
    tasks: List[TaskResponseSchema] = Field(default_factory=list)

class AgentPublishSchema(BaseModel):
    version: Optional[str] = Field(None, pattern=r"^\d+\.\d+\.\d+$")
    notes: Optional[str] = None

# Bulk operation schemas
class AgentImportSchema(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    instructions: Optional[str] = None
    capabilities: Optional[List[str]] = Field(default_factory=list)
    tools: List[ToolCreateSchema] = Field(default_factory=list)
    tasks: List[TaskCreateSchema] = Field(default_factory=list)

# Statistics schemas
class AgentStatsSchema(BaseModel):
    total: int
    published: int
    draft: int
    deprecated: int = 0

class ToolStatsSchema(BaseModel):
    total: int
    enabled: int = 0
    disabled: int = 0

class TaskStatsSchema(BaseModel):
    total: int
    pending: int
    in_progress: int
    completed: int
    failed: int

class SystemStatsSchema(BaseModel):
    agents: AgentStatsSchema
    tools: ToolStatsSchema
    tasks: TaskStatsSchema
    last_updated: datetime

# Error schemas
class ErrorSchema(BaseModel):
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Response schemas for API documentation
class SuccessMessageSchema(BaseModel):
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthSchema(BaseModel):
    status: str
    timestamp: datetime
    version: str
