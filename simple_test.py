"""
Simple test to verify API is working
"""

import requests
import json

def test_health():
    """Test if API is running"""
    try:
        response = requests.get("http://127.0.0.1:8000/health")
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_create_simple_agent():
    """Create a simple agent"""
    try:
        agent_data = {
            "name": "Test Weather Agent",
            "description": "A simple test agent for weather information",
            "capabilities": ["weather_forecast"]
        }
        
        response = requests.post("http://127.0.0.1:8000/agents", json=agent_data)
        print(f"Create Agent Status: {response.status_code}")
        
        if response.status_code == 201:
            agent = response.json()
            print(f"Created Agent ID: {agent['id']}")
            print(f"Agent Name: {agent['name']}")
            return agent['id']
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_list_agents():
    """List all agents"""
    try:
        response = requests.get("http://127.0.0.1:8000/agents")
        print(f"List Agents Status: {response.status_code}")
        
        if response.status_code == 200:
            agents = response.json()
            print(f"Total Agents: {len(agents)}")
            for agent in agents:
                print(f"  - {agent['name']} (ID: {agent['id']}, Status: {agent['status']})")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("SIMPLE API TEST")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    if not test_health():
        print("API is not running. Please start it first.")
        exit(1)
    
    # Test 2: Create agent
    print("\n2. Creating a test agent...")
    agent_id = test_create_simple_agent()
    
    # Test 3: List agents
    print("\n3. Listing all agents...")
    test_list_agents()
    
    print("\n" + "=" * 50)
    print("TEST COMPLETED")
    print("=" * 50)
    print("\nYou can now:")
    print("1. Open http://127.0.0.1:8000/docs for interactive API documentation")
    print("2. Open http://127.0.0.1:8000/stats for system statistics")
    print("3. Use the API endpoints to manage agents, tools, and tasks")
