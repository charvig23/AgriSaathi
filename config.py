"""
Environment configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/agrisathi")

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# API configuration
API_V1_STR = "/api/v1"
PROJECT_NAME = "AgriSaathi Agent Management API"
VERSION = "2.0.0"
DESCRIPTION = "Comprehensive CRUD system for managing AI agents, tools, and tasks"

# CORS configuration
ALLOWED_HOSTS = ["*"]  # Configure appropriately for production
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# Database connection configuration
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_ECHO = os.getenv("DB_ECHO", "False").lower() == "true"

# FAISS configuration
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
