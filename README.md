## DOOSAN Risk Management AI

An AI-powered safety assistant for accident analysis, risk assessments, chemical safety lookups, and regulatory guidance. The project provides:

- A Streamlit chat app for interactive Q&A with tool-use
- A FastAPI server that reuses the same logic to expose REST endpoints (no code duplication)

### Features
- Accident records retrieval with contextual compression and reranking
- Risk assessment insights and structured risk tables (JSON)
- Chemical identification and safety info via KOSHA MSDS open API
- Regulations and compliance guidance
- Dynamic risk assessment using frequency × severity model
- Bilingual support (English/Korean) based on the user query language

---

## Project Structure
- `streamlit_app.py`: Main app, tools, and output generators
- `fastapi_server.py`: API server importing all functionality from `streamlit_app.py`
- `requirements.txt`: Full dependency set for Streamlit app
- `requirements_fastapi.txt`: Minimal dependencies for FastAPI server
- `README_FastAPI.md`: API-focused quickstart (kept for reference)

---

## Requirements
Python 3.10+
langchain>=0.3.27
langchain-openai>=0.3.32
langchain-qdrant>=0.2.0
python-dotenv>=1.1.1
qdrant-client[fastembed]>=1.15.1
streamlit>=1.49.1
tabulate==0.9.0
Recommended to use a virtual environment.

### Environment Variables
Create a `.env` file in the project root with the following keys as needed:

```env
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
KOSHA_API_KEY=your_kosha_api_key
# Optional if you plan to use Anthropic locally in the future
# ANTHROPIC_API_KEY=your_anthropic_api_key
```

Notes:
- The app uses OpenAI (chat and JSON responses), Cohere (Rerank v3.5), and Qdrant (vector store).
- KOSHA API is used to fetch MSDS data for chemical lookups.

---

## Installation

From the project root:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If you only want to run the FastAPI server, you can instead install:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements_fastapi.txt
```

---

## Run the Streamlit App

```bash
.\.venv\Scripts\activate
streamlit run streamlit_app.py
```

You will see a chat interface titled “DOOSAN RISK MANAGEMENT AI Chatbot 📄”. The assistant will automatically pick the right tool based on your query. If you ask for a “risk assessment table” (or "테이블"), it will render a data table derived from JSON.

---

## Run the FastAPI Server

The API server reuses/imports all logic from `streamlit_app.py` (no duplication).

### Development (auto-reload)
```bash
.\.venv\Scripts\activate
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload
```

### Production (example)
```bash
gunicorn fastapi_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Health and Docs
- API Root: `http://localhost:8000/`
- Health: `http://localhost:8000/health`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Overview

### Main Query
- POST `/query`
  - Body: `{ "query": "your question" }`
  - The backend decides which tool to call and returns a final AI response.
  - Response model includes `result`, `tool_used`, `success`, and `chat_history`.

### Risk Assessment Table (JSON)
- POST `/risk-assessment-table`
  - Body: `{ "query": "risk assessment table for <topic>" }`
  - Returns structured JSON array suitable for tabular display.

### Individual Tool Endpoints
- POST `/tools/accident-records`
- POST `/tools/chemical-usage`
- POST `/tools/risk-assessment`
- POST `/tools/regulations`
- POST `/tools/chemical-details`
- POST `/tools/dynamic-risk-assessment`

All accept: `{ "query": "..." }` and return `{ data, success, message, tool_name }`.

### Example Requests

```bash
# Main AI query (tool auto-selected)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the main safety hazards in welding operations?"}'

# Risk assessment table (JSON)
curl -X POST "http://localhost:8000/risk-assessment-table" \
  -H "Content-Type: application/json" \
  -d '{"query":"Show me risk assessment table for welding operations"}'

# Chemical details via KOSHA (identified from natural-language query)
curl -X POST "http://localhost:8000/tools/chemical-details" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the CAS number of oxygen?"}'
```

---

## How It Works (High Level)

The app defines six tools in `streamlit_app.py` using LangChain tools:
- `get_accident_records`, `get_chemical_usage`, `get_risk_assessment`, `get_regulations_data`, `get_chemical_details`, `dynamic_risk_assessment`

Each tool retrieves domain data (Qdrant for vector search; KOSHA for MSDS) and the corresponding output functions format the final response with clear, structured sections. The language of responses automatically follows the language of the user’s query (English or Korean).

The FastAPI server imports these same functions and exposes REST endpoints without duplicating logic.

---

## Troubleshooting
- Ensure `.env` is present and keys are valid.
- Qdrant collections referenced in code must exist and be reachable using `QDRANT_URL` and `QDRANT_API_KEY`.
- If JSON rendering for the risk table fails, the API falls back to a safe structure with fields set to "N/A".
- For Windows PowerShell, use the provided venv activation path: ` .\.venv\Scripts\activate `.

---

## License
Proprietary – Internal use only unless otherwise specified by the project owner.


