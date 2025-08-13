# AgriSaathi Agent Management System

A comprehensive CRUD system for managing AI agents, their tools, and tasks with publishing capabilities.

## Features

- **Agent Management**: Create, update, publish, and manage AI agents
- **Tool Management**: Define and manage tools for each agent with input/output schemas
- **Task Management**: Create and track tasks with status management
- **Publishing System**: Publish/unpublish agents with versioning
- **Schema Validation**: Comprehensive input/output schema validation
- **Bulk Operations**: Import/export agents with all associated data
- **Statistics**: Real-time system statistics and health monitoring

## API Endpoints

### Agents
- `POST /agents` - Create new agent
- `GET /agents` - List all agents (with filtering)
- `GET /agents/{agent_id}` - Get specific agent
- `PUT /agents/{agent_id}` - Update agent
- `POST /agents/{agent_id}/publish` - Publish agent
- `POST /agents/{agent_id}/unpublish` - Unpublish agent
- `DELETE /agents/{agent_id}` - Delete agent
- `GET /agents/{agent_id}/export` - Export agent data
- `POST /agents/import` - Import agent data

### Tools
- `POST /agents/{agent_id}/tools` - Create tool for agent
- `GET /agents/{agent_id}/tools` - Get all tools for agent
- `GET /tools/{tool_id}` - Get specific tool
- `PUT /agents/{agent_id}/tools/{tool_id}` - Update tool
- `DELETE /agents/{agent_id}/tools/{tool_id}` - Delete tool

### Tasks
- `POST /agents/{agent_id}/tasks` - Create task for agent
- `GET /agents/{agent_id}/tasks` - Get all tasks for agent
- `GET /tasks/{task_id}` - Get specific task
- `PUT /agents/{agent_id}/tasks/{task_id}` - Update task
- `DELETE /agents/{agent_id}/tasks/{task_id}` - Delete task

### System
- `GET /health` - Health check
- `GET /stats` - System statistics
- `GET /` - API information

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up PostgreSQL database and update the connection string in `agents_api_db.py`

3. Run the application:
```bash
uvicorn agents_api_db:app --reload
```

## Agent Schema Example

```json
{
  "name": "Weather Assistant",
  "description": "An agent that provides weather information",
  "version": "1.0.0",
  "status": "draft",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {"type": "string"},
      "date": {"type": "string", "format": "date"}
    },
    "required": ["location"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "temperature": {"type": "number"},
      "conditions": {"type": "string"},
      "humidity": {"type": "number"}
    }
  },
  "system_prompt": "You are a helpful weather assistant.",
  "capabilities": ["weather_forecast", "weather_alerts"]
}
```

## Tool Schema Example

```json
{
  "name": "Weather API Tool",
  "description": "Fetches weather data from external API",
  "tool_type": "api",
  "config": {
    "api_url": "https://api.weather.com/v1",
    "auth_type": "api_key"
  },
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {"type": "string"},
      "units": {"type": "string", "enum": ["metric", "imperial"]}
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "temperature": {"type": "number"},
      "conditions": {"type": "string"}
    }
  },
  "implementation": "def fetch_weather(location, units='metric'): ..."
}
```

## Task Schema Example

```json
{
  "name": "Get Weather for Delhi",
  "description": "Fetch current weather conditions for Delhi",
  "priority": 5,
  "input_data": {
    "location": "Delhi, India",
    "units": "metric"
  },
  "assigned_tool_id": "tool-id-here"
}
```

## Status Management

### Agent Status
- `draft` - Agent is in development
- `published` - Agent is live and available
- `deprecated` - Agent is no longer recommended

### Task Status
- `pending` - Task is waiting to be processed
- `in_progress` - Task is currently being executed
- `completed` - Task has been successfully completed
- `failed` - Task execution failed

### Tool Types
- `api` - External API integration
- `database` - Database query tool
- `llm` - Language model tool
- `custom` - Custom implementation

## Environment Variables

Create a `.env` file with:
```
DATABASE_URL=postgresql://user:password@localhost:5432/agrisathi
SECRET_KEY=your-secret-key-here
```
