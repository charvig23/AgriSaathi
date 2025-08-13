"""
SQLAlchemy ORM models for AgriSaathi Agent Management System
"""

from sqlalchemy import Column, String, Text, JSON, ForeignKey, Boolean, DateTime, Integer
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base
from schemas import AgentStatus, TaskStatus, ToolType

class Agent(Base):
    """
    Agent model representing an AI agent with its configuration and metadata
    """
    __tablename__ = "agents"
    
    # Primary fields
    id = Column(String, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    version = Column(String(20), default="1.0.0")
    status = Column(String(20), default=AgentStatus.DRAFT)
    
    # Schema definitions
    input_schema = Column(JSON)  # JSON schema for agent inputs
    output_schema = Column(JSON)  # JSON schema for agent outputs
    
    # Configuration
    system_prompt = Column(Text)
    instructions = Column(Text)
    capabilities = Column(JSON)  # List of capabilities
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime, nullable=True)
    
    # Relationships
    tools = relationship(
        "Tool", 
        back_populates="agent", 
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    tasks = relationship(
        "Task", 
        back_populates="agent", 
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    def __repr__(self):
        return f"<Agent(id={self.id}, name={self.name}, status={self.status})>"

class Tool(Base):
    """
    Tool model representing a tool that can be used by an agent
    """
    __tablename__ = "tools"
    
    # Primary fields
    id = Column(String, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    tool_type = Column(String(20), default=ToolType.CUSTOM)
    
    # Configuration and schemas
    config = Column(JSON)  # Tool configuration
    input_schema = Column(JSON)  # JSON schema for tool inputs
    output_schema = Column(JSON)  # JSON schema for tool outputs
    implementation = Column(Text)  # Code/implementation details
    
    # Status
    is_enabled = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Key
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    
    # Relationships
    agent = relationship("Agent", back_populates="tools")
    
    def __repr__(self):
        return f"<Tool(id={self.id}, name={self.name}, type={self.tool_type})>"

class Task(Base):
    """
    Task model representing a task that can be executed by an agent
    """
    __tablename__ = "tasks"
    
    # Primary fields
    id = Column(String, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(20), default=TaskStatus.PENDING)
    priority = Column(Integer, default=5)  # 1-10 priority scale
    
    # Data
    input_data = Column(JSON)  # Input data for the task
    output_data = Column(JSON)  # Output data from the task
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Foreign Keys
    assigned_tool_id = Column(String, ForeignKey("tools.id"), nullable=True)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    
    # Relationships
    agent = relationship("Agent", back_populates="tasks")
    
    def __repr__(self):
        return f"<Task(id={self.id}, name={self.name}, status={self.status})>"
