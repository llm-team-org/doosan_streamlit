# fastapi_server.py
import os
import random
import asyncio
from enum import IntEnum
from typing import Optional, List, Dict, Tuple, Callable, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from patch_url_function import patch_url_function_progress
from pydantic_models import patch_response_posco, ResponseStatus

# ===== Load env (keys for your tools' internals) =====
load_dotenv()

# ===== Import your existing tool + output functions (from your Streamlit code) =====
# IMPORTANT: ensure any Streamlit UI code in streamlit_app.py is guarded by:
# if __name__ == "__main__":  ... UI code ...
from streamlit_app import (
    # Tools
    get_accident_records,
    get_chemical_usage,
    get_risk_assessment,
    get_regulations_data,
    get_chemical_details,
    dynamic_risk_assessment,
    # Output builders
    accident_output,
    risk_assessment_output,
    chemical_output,
    regulations_output,
    dynamic_risk_assessment_output,
)

# ===== FastAPI app =====
app = FastAPI(
    title="DOOSAN Risk Management AI API",
    description="Single-endpoint dispatcher that routes type=1..6 to the right function.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Enum (exactly as you provided) =====
class QueryType(IntEnum):
    CHEMICAL = 1
    ACCIDENT_RECORDS = 2
    RISK_ASSESSMENT = 3
    REGULATIONS_OUTPUT = 4
    DYNAMIC_RISK_ASSESSMENT = 5
    DATASETS_OR_KOSHA = 6

# ===== Request / Response models =====
class UnifiedQueryRequest(BaseModel):
    request_id: int
    query: str
    type: Optional[int] =None
    patch_url: Optional[str] = None

class UnifiedQueryResponse(BaseModel):
    request_id: int
    query: str
    type: Optional[int] =None
    response_id: int
    status: int            # 1 = success, 0 = failure
    answer: str
    sources: List[int] = []  # optional numeric sources if you later surface them

# ===== Helpers =====
def extract_sources(output: Any) -> List[int]:
    """
    Best-effort numeric source extraction. Right now your builders return strings,
    so this will usually return []. If you later return dict/list with 'sources',
    this will pick numeric ids out.
    """
    try:
        if output is None:
            return []
        s = output
        if isinstance(output, dict):
            s = output.get("sources") or output.get("SOURCES") or []
        out: List[int] = []
        if isinstance(s, list):
            for item in s:
                if isinstance(item, int):
                    out.append(item)
                elif isinstance(item, dict):
                    if isinstance(item.get("id"), int):
                        out.append(item["id"])
                    elif isinstance(item.get("source_id"), int):
                        out.append(item["source_id"])
        return out
    except Exception:
        return []

# Thin wrappers: one tool call → one output builder → string answer
def _build_chemical(q: str) -> str:
    # type 1: CHEMICAL → get_chemical_details → chemical_output
    table_data = get_chemical_details(q)
    return chemical_output(table_data=table_data, query=q)

def _build_accident(q: str) -> str:
    # type 2: ACCIDENT_RECORDS → get_accident_records → accident_output
    docs = get_accident_records(q)
    return accident_output(accident_docs=docs, query=q)

def _build_risk(q: str) -> str:
    # type 3: RISK_ASSESSMENT → get_risk_assessment → risk_assessment_output
    docs = get_risk_assessment(q)
    return risk_assessment_output(risk_assessment_docs=docs, query=q)

def _build_regulations(q: str) -> str:
    # type 4: REGULATIONS_OUTPUT → get_regulations_data → regulations_output
    docs = get_regulations_data(q)
    return regulations_output(regulations_docs=docs, query=q)

def _build_dynamic(q: str) -> str:
    # type 5: DYNAMIC_RISK_ASSESSMENT → dynamic_risk_assessment → dynamic_risk_assessment_output
    docs = dynamic_risk_assessment(q)
    return dynamic_risk_assessment_output(risk_assessment_docs=docs, query=q)

def _build_datasets_or_kosha(q: str) -> str:
    # type 6: DATASETS_OR_KOSHA → get_chemical_usage (your dataset path) → chemical_output
    docs = get_chemical_usage(q)
    return chemical_output(table_data=docs, query=q)

# Final routing table used by the endpoint
TOOL_MAP: Dict[int, Tuple[str, Callable[[str], str]]] = {
    int(QueryType.CHEMICAL): ("get_chemical_details", _build_chemical),
    int(QueryType.ACCIDENT_RECORDS): ("get_accident_records", _build_accident),
    int(QueryType.RISK_ASSESSMENT): ("get_risk_assessment", _build_risk),
    int(QueryType.REGULATIONS_OUTPUT): ("get_regulations_data", _build_regulations),
    int(QueryType.DYNAMIC_RISK_ASSESSMENT): ("dynamic_risk_assessment", _build_dynamic),
    int(QueryType.DATASETS_OR_KOSHA): ("get_chemical_usage", _build_datasets_or_kosha),
}

# ===== Basic health + root =====
@app.get("/")
def root():
    return {"message": "DOOSAN Risk Management AI API (unified)", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "message": "API is running"}

# ===== Single endpoint: /genai/doosan_chatbot/ =====
@app.post("/genai/doosan_chatbot/", response_model=UnifiedQueryResponse)
async def doosan_chatbot(req: UnifiedQueryRequest):
    response_id = random.randint(100_000, 999_999_999)

    try:
        # Optional: notify PROCESSING state
        if req.patch_url:
            await patch_url_function_progress(
                req.patch_url,
                patch_response_posco(id=str(req.request_id), response=None, status=ResponseStatus.PROCESSING),
            )

        # validate type
        if req.type not in TOOL_MAP:
            msg = f"Unsupported type {req.type}. Expected 1..6."
            if req.patch_url:
                await patch_url_function_progress(
                    req.patch_url,
                    patch_response_posco(id=str(req.request_id), response=msg, status=ResponseStatus.FAILED),
                )
            return UnifiedQueryResponse(
                request_id=req.request_id,
                query=req.query,
                type=req.type,
                response_id=response_id,
                status=0,
                answer=msg,
                sources=[],
            )

        tool_used, builder = TOOL_MAP[req.type]
        # one call → mapped function
        answer: str = builder(req.query)

        # Optional: notify SUCCESS state
        if req.patch_url:
            await patch_url_function_progress(
                req.patch_url,
                patch_response_posco(id=str(req.request_id), response=answer, status=ResponseStatus.SUCCESS),
            )

        # extract numeric sources if you ever return them
        sources = extract_sources(answer) if not isinstance(answer, str) else []

        return UnifiedQueryResponse(
            request_id=req.request_id,
            query=req.query,
            type=req.type,
            response_id=response_id,
            status=1,
            answer=answer,
            sources=sources,
        )

    except Exception as e:
        # notify EXCEPTION state if possible
        try:
            if 'req' in locals() and getattr(req, 'patch_url', None):
                await patch_url_function_progress(
                    req.patch_url,
                    patch_response_posco(id=str(req.request_id), response=str(e), status=ResponseStatus.EXCEPTION),
                )
        finally:
            # keep it simple; return failure with message
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ===== Run locally =====
if __name__ == "__main__":
    uvicorn.run(
        app,  # run the FastAPI app directly
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False
    )
