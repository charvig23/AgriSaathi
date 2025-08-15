# AgriSaathi Setup Guide

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd AgriSaathi
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv agrisaathi_env
agrisaathi_env\Scripts\activate

# Linux/Mac
python -m venv agrisaathi_env
source agrisaathi_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your actual API keys
# DATABASE_URL=your_postgresql_url
# OPENAI_API_KEY=your_openai_key
# GOOGLE_API_KEY=your_google_key
# HUGGING_FACE_API_KEY=your_hf_key
```

### 5. Run the Application
```bash
uvicorn agents_api_db:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Access the API
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 📋 Requirements

- Python 3.8+
- PostgreSQL database
- API keys for LLM services (optional)

## 🔧 Project Structure

```
AgriSaathi/
├── agents_api_db.py    # Main FastAPI application
├── database.py         # Database connection
├── models.py          # SQLAlchemy ORM models
├── schemas.py         # Pydantic schemas
├── requirements.txt   # Dependencies
├── .env              # Environment variables
└── README.md         # Documentation
```

## 🌟 Features

- ✅ Complete CRUD API for agents, tools, and tasks
- ✅ LLM-powered intelligent tool selection
- ✅ Support for multiple LLM providers (OpenAI, Google, local models)
- ✅ PostgreSQL database with advanced schemas
- ✅ Interactive API documentation
- ✅ Comprehensive error handling
