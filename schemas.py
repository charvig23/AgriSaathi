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

class LLMModel(str, Enum):
    # OpenAI GPT Models (Free tier available)
    GPT_4O_MINI = "gpt-4o-mini"  # Most cost-effective GPT-4 class model
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4O = "gpt-4o"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    
    # Google Models (Free tier available)
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    
    # Local/Open Source Models (Free with local hosting)
    LLAMA2_7B = "llama2-7b"
    LLAMA2_13B = "llama2-13b"
    LLAMA2_70B = "llama2-70b"
    MISTRAL_7B = "mistral-7b"
    MIXTRAL_8X7B = "mixtral-8x7b"
    GEMMA_2B = "gemma-2b"
    GEMMA_7B = "gemma-7b"
    CODELLAMA_7B = "codellama-7b"
    OPENCHAT_7B = "openchat-7b"
    
    # Default/None option
    NONE = "none"

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
    
    # LLM Configuration for intelligent tool selection
    llm_model: LLMModel = LLMModel.NONE
    llm_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tool_selection_prompt: Optional[str] = Field(
        default="Based on the user query and available tools, select the most appropriate tool to use. "
                "Consider the tool descriptions and capabilities."
    )
    enable_intelligent_routing: bool = Field(default=False)

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
    
    # LLM Configuration updates
    llm_model: Optional[LLMModel] = None
    llm_config: Optional[Dict[str, Any]] = None
    tool_selection_prompt: Optional[str] = None
    enable_intelligent_routing: Optional[bool] = None

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
    
    # LLM Configuration in response
    llm_model: LLMModel
    llm_config: Optional[Dict[str, Any]]
    tool_selection_prompt: Optional[str]
    enable_intelligent_routing: bool
    
    tools: List[ToolResponseSchema] = Field(default_factory=list)
    tasks: List[TaskResponseSchema] = Field(default_factory=list)

class AgentPublishSchema(BaseModel):
    version: Optional[str] = Field(None, pattern=r"^\d+\.\d+\.\d+$")
    notes: Optional[str] = None

# Tool Selection schemas for LLM-based routing
class ToolSelectionRequest(BaseModel):
    user_query: str = Field(..., min_length=1, description="The user's query or request")
    context: Optional[str] = Field(None, description="Additional context for tool selection")
    
class ToolSelectionResponse(BaseModel):
    selected_tool_id: str = Field(..., description="ID of the selected tool")
    selected_tool_name: str = Field(..., description="Name of the selected tool")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in tool selection")
    reasoning: str = Field(..., description="Explanation for why this tool was selected")
    fallback_tools: Optional[List[str]] = Field(default_factory=list, description="Alternative tool IDs if primary fails")

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
    
    # LLM Configuration for imports
    llm_model: LLMModel = LLMModel.NONE
    llm_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tool_selection_prompt: Optional[str] = None
    enable_intelligent_routing: bool = Field(default=False)
    
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
