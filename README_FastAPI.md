# DOOSAN Risk Management AI - FastAPI Server

A clean and efficient FastAPI server that imports all functionality from the Streamlit app, providing the same risk management capabilities through a REST API.

## ✨ Key Features

- **Zero Code Duplication**: Imports all functions directly from `streamlit_app.py`
- **Same AI Tools**: All 6 tools including the new `dynamic_risk_assessment`
- **REST API Interface**: Clean endpoints for easy integration
- **Auto Documentation**: Swagger UI and ReDoc included
- **CORS Support**: Ready for web applications

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_fastapi.txt
```

### 2. Set Environment Variables
Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
KOSHA_API_KEY=your_kosha_api_key
```

### 3. Run the Server
```bash
python run_fastapi.py
```

Or directly:
```bash
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload
```

## 📊 API Endpoints

### Main Query Endpoint
- **POST** `/query` - Universal endpoint using AI tools (same as Streamlit chat)
  - AI automatically selects the appropriate tool based on your query
  - Supports all 6 tools: accident records, chemical usage, risk assessment, regulations, dynamic risk assessment, chemical details

### Specialized Endpoints
- **POST** `/risk-assessment-table` - Risk assessment table data (JSON format)

### Utility Endpoints
- **GET** `/` - API information
- **GET** `/health` - Health check

## 🌐 Access Points

- **API Server**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📝 Example Usage

### Using curl:
```bash
# Main query endpoint (AI selects appropriate tool)
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the safety risks in chemical handling?"}'

# Risk assessment table data
curl -X POST "http://localhost:8000/risk-assessment-table" \
     -H "Content-Type: application/json" \
     -d '{"query": "Show me risk assessment table for welding operations"}'

# Chemical details (handled by main query endpoint)
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the CAS number of oxygen?"}'
```

### Using Python requests:
```python
import requests

# Query the API
response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What are the main safety hazards in our facility?"}
)

result = response.json()
print(result["result"])
```

## 🔧 Architecture

The FastAPI server is designed to be lightweight and maintainable:

- **Imports**: All functions imported from `streamlit_app.py`
- **No Duplication**: Zero code duplication between Streamlit and FastAPI
- **Same Logic**: Identical AI processing and tool selection
- **Clean API**: RESTful endpoints with proper error handling

## 📋 Response Format

### Query Response
```json
{
  "result": "AI-generated response text",
  "tool_used": "tool_name",
  "success": true
}
```

### Risk Assessment Table Response
```json
{
  "data": [
    {
      "Task/Process Name": "Welding Operations",
      "Hazard Factor": "Fire and Explosion",
      "Current Risk Level": "Frequency (3) x Severity (4) = Risk Score (12)",
      // ... more fields
    }
  ],
  "success": true,
  "message": "Retrieved 1 risk assessment records"
}
```

## 🛠️ Development

The server automatically reloads when you make changes to the code:
```bash
uvicorn fastapi_server:app --reload --host 0.0.0.0 --port 8000
```

## 🚀 Production Deployment

For production, use a production ASGI server:
```bash
gunicorn fastapi_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 🔄 Maintenance

Since all functions are imported from `streamlit_app.py`, any updates to the Streamlit app will automatically be available in the FastAPI server without any code changes needed in the FastAPI files.
