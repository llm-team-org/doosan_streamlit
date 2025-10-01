# pylint: disable=all
# fmt: off
# flake8: noqaimport os

import os
from functions import (accident_output,get_accident_records,get_chemical_details,chemical_output,get_risk_assessment,
                       risk_assessment_output,regulations_output,get_regulations_data,dynamic_risk_assessment,dynamic_risk_assessment_output,
                       get_chemical_usage)

import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from patch_url_function import patch_url_function_progress
from pydantic_models import UnifiedQueryResponse,UnifiedQueryRequest

# ===== Load env (keys for your tools' internals) =====
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")+"/api/documentation"


# ===== FastAPI app =====
app = FastAPI(
    title="DOOSAN Risk Management AI API",
    description="Single-endpoint dispatcher that routes type=1..6 to the right function.",
    version="2.0.0",
)

type_id=[1,2,3,4,5,6]

# ===== Basic health + root =====
@app.get("/")
def root():
    return {"message": "DOOSAN Risk Management AI API (unified)", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "message": "API is running"}

# ===== Single endpoint: /genai/doosan_chatbot/ =====
@app.post("/genai/doosan_chatbot/")
async def doosan_chatbot(req: UnifiedQueryRequest):
    try:
        # Api start
        await patch_url_function_progress(
            url=BACKEND_URL,
            response_data=UnifiedQueryResponse(request_id=req.request_id, status=1)
        )
        # validate type
        user_type=req.type
        if user_type in type_id:
            if user_type ==2:
                records= get_accident_records(query=req.query)
                answer=accident_output(accident_docs=records,query=req.query)
                await patch_url_function_progress(
                    url=BACKEND_URL,
                    response_data=UnifiedQueryResponse(request_id=req.request_id, status=2,answer=answer)
                )
            elif user_type ==1:
                docs= get_chemical_usage(query=req.query)
                answer=chemical_output(table_data=docs,query=req.query)
                await patch_url_function_progress(
                    url=BACKEND_URL,
                    response_data=UnifiedQueryResponse(request_id=req.request_id, status=2,answer=answer)
                )
            elif user_type ==3:
                docs= get_risk_assessment(query=req.query)
                answer=risk_assessment_output(risk_assessment_docs=docs,query=req.query)
                await patch_url_function_progress(
                    url=BACKEND_URL,
                    response_data=UnifiedQueryResponse(request_id=req.request_id, status=2,answer=answer)
                )
            elif user_type ==4:
                docs=get_regulations_data(query=req.query)
                answer=regulations_output(regulations_docs=docs,query=req.query)
                await patch_url_function_progress(
                    url=BACKEND_URL,
                    response_data=UnifiedQueryResponse(request_id=req.request_id, status=2,answer=answer)
                )
            elif user_type ==5:
                docs=dynamic_risk_assessment(query=req.query)
                answer=dynamic_risk_assessment_output(risk_assessment_docs=docs,query=req.query)
                await patch_url_function_progress(
                    url=BACKEND_URL,
                    response_data=UnifiedQueryResponse(request_id=req.request_id, status=2, answer=answer)
                )
            elif user_type ==6:
                table_data = get_chemical_details(query=req.query)
                answer = chemical_output(table_data=table_data, query=req.query)
                await patch_url_function_progress(
                    url=BACKEND_URL,
                    response_data=UnifiedQueryResponse(request_id=req.request_id, status=2, answer=answer)
                )
            else:
                await patch_url_function_progress(
                    url=BACKEND_URL,
                    response_data=UnifiedQueryResponse(request_id=req.request_id, status=3)
                )

    except Exception as e:
        await patch_url_function_progress(
            url=BACKEND_URL,
            response_data=UnifiedQueryResponse(request_id=req.request_id, status=3)
        )
        raise e


# ===== Run locally =====
if __name__ == "__main__":
    uvicorn.run(
        app,  # run the FastAPI app directly
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False
    )
