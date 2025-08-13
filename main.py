"""
Startup script for AgriSaathi Agent Management API
"""

import uvicorn
from config import PROJECT_NAME, VERSION

if __name__ == "__main__":
    print(f"Starting {PROJECT_NAME} v{VERSION}")
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
