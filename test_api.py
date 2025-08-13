"""
Test examples for AgriSaathi Agent Management API
"""

import requests
import json
from datetime import datetime

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_create_agent():
    """Test creating a new agent"""
    agent_data = {
        "name": "Weather Assistant",
        "description": "An AI agent that provides weather information and forecasts",
        "version": "1.0.0",
        "status": "draft",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location for weather information"
                },
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Date for forecast (optional)"
                }
            },
            "required": ["location"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "conditions": {"type": "string"},
                "humidity": {"type": "number"},
                "forecast": {"type": "array"}
            }
        },
        "system_prompt": "You are a helpful weather assistant that provides accurate and up-to-date weather information.",
        "instructions": "Always provide temperature in both Celsius and Fahrenheit. Include humidity and any weather alerts.",
        "capabilities": ["weather_forecast", "weather_alerts", "historical_weather"]
    }
    
    response = requests.post(f"{BASE_URL}/agents", json=agent_data)
    print(f"Create Agent Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json().get("id")

def test_create_tool(agent_id):
    """Test creating a tool for an agent"""
    tool_data = {
        "name": "Weather API Client",
        "description": "Tool to fetch weather data from external weather API",
        "tool_type": "api",
        "config": {
            "api_url": "https://api.openweathermap.org/data/2.5",
            "auth_type": "api_key",
            "timeout": 30,
            "retry_attempts": 3
        },
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string", "enum": ["metric", "imperial", "kelvin"]},
                "lang": {"type": "string", "default": "en"}
            },
            "required": ["location"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "feels_like": {"type": "number"},
                "humidity": {"type": "number"},
                "pressure": {"type": "number"},
                "visibility": {"type": "number"},
                "weather": {
                    "type": "object",
                    "properties": {
                        "main": {"type": "string"},
                        "description": {"type": "string"}
                    }
                }
            }
        },
        "implementation": """
import requests

def fetch_weather(location, units='metric', lang='en'):
    api_key = os.getenv('WEATHER_API_KEY')
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': location,
        'appid': api_key,
        'units': units,
        'lang': lang
    }
    response = requests.get(url, params=params)
    return response.json()
        """,
        "is_enabled": True
    }
    
    response = requests.post(f"{BASE_URL}/agents/{agent_id}/tools", json=tool_data)
    print(f"Create Tool Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json().get("id")

def test_create_task(agent_id, tool_id):
    """Test creating a task for an agent"""
    task_data = {
        "name": "Get Weather for Delhi",
        "description": "Fetch current weather conditions for Delhi, India",
        "priority": 7,
        "input_data": {
            "location": "Delhi, India",
            "units": "metric",
            "lang": "en"
        },
        "assigned_tool_id": tool_id
    }
    
    response = requests.post(f"{BASE_URL}/agents/{agent_id}/tasks", json=task_data)
    print(f"Create Task Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json().get("id")

def test_publish_agent(agent_id):
    """Test publishing an agent"""
    publish_data = {
        "version": "1.0.1",
        "notes": "Initial release of weather assistant agent"
    }
    
    response = requests.post(f"{BASE_URL}/agents/{agent_id}/publish", json=publish_data)
    print(f"Publish Agent Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_list_agents():
    """Test listing all agents"""
    response = requests.get(f"{BASE_URL}/agents")
    print(f"List Agents Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_get_statistics():
    """Test getting system statistics"""
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Statistics Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_export_agent(agent_id):
    """Test exporting an agent"""
    response = requests.get(f"{BASE_URL}/agents/{agent_id}/export")
    print(f"Export Agent Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_complete_workflow():
    """Test the complete workflow"""
    print("=" * 60)
    print("TESTING COMPLETE AGENT MANAGEMENT WORKFLOW")
    print("=" * 60)
    
    # 1. Create an agent
    print("\n1. Creating an agent...")
    agent_id = test_create_agent()
    
    if not agent_id:
        print("Failed to create agent, stopping tests")
        return
    
    # 2. Create a tool for the agent
    print(f"\n2. Creating a tool for agent {agent_id}...")
    tool_id = test_create_tool(agent_id)
    
    # 3. Create a task for the agent
    print(f"\n3. Creating a task for agent {agent_id}...")
    task_id = test_create_task(agent_id, tool_id)
    
    # 4. Publish the agent
    print(f"\n4. Publishing agent {agent_id}...")
    test_publish_agent(agent_id)
    
    # 5. List all agents
    print("\n5. Listing all agents...")
    test_list_agents()
    
    # 6. Get system statistics
    print("\n6. Getting system statistics...")
    test_get_statistics()
    
    # 7. Export the agent
    print(f"\n7. Exporting agent {agent_id}...")
    test_export_agent(agent_id)
    
    print("\n" + "=" * 60)
    print("WORKFLOW TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    # Check if the API is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("API is running, starting tests...")
            test_complete_workflow()
        else:
            print("API is not responding correctly")
    except requests.exceptions.ConnectionError:
        print("Cannot connect to API. Make sure it's running on http://localhost:8000")
        print("Start the API with: python main.py")
