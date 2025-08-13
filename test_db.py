"""
Test script to create an agent and tool in PostgreSQL database
"""
import requests
import json

def test_db_connection():
    """Test if we can create an agent and tool"""
    
    # 1. Create an Agent
    agent_data = {
        "name": "Weather Agent",
        "description": "Agent for weather forecasting and agricultural advice",
        "version": "1.0.0",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "date": {"type": "string", "format": "date"}
            }
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "forecast": {"type": "string"},
                "recommendations": {"type": "array"}
            }
        },
        "system_prompt": "You are a weather expert helping farmers.",
        "capabilities": ["weather_forecast", "agricultural_advice"]
    }
    
    print("\nCreating agent...")
    response = requests.post(
        "http://127.0.0.1:8000/agents",
        json=agent_data
    )
    
    if response.status_code == 201:
        agent = response.json()
        print(f"✅ Agent created successfully!")
        print(f"Agent ID: {agent['id']}")
        print(f"Agent Name: {agent['name']}")
        
        # 2. Create a Tool for this Agent
        tool_data = {
            "name": "IMD Weather API",
            "description": "Fetches weather data from IMD",
            "tool_type": "api",
            "config": {
                "api_url": "https://api.imd.gov.in/weather",
                "method": "GET"
            },
            "input_schema": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"}
                }
            }
        }
        
        print("\nCreating tool for the agent...")
        response = requests.post(
            f"http://127.0.0.1:8000/agents/{agent['id']}/tools",
            json=tool_data
        )
        
        if response.status_code == 201:
            tool = response.json()
            print("✅ Tool created successfully!")
            print(f"Tool ID: {tool['id']}")
            print(f"Tool Name: {tool['name']}")
        else:
            print(f"❌ Failed to create tool: {response.text}")
    else:
        print(f"❌ Failed to create agent: {response.text}")

if __name__ == "__main__":
    print("Testing PostgreSQL Database Connection...")
    print("-" * 50)
    test_db_connection()
