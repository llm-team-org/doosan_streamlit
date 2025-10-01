import os
import uvicorn
import aiohttp
import psutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import cohere

# Import all functions from streamlit_app
from streamlit_app import (
    # Tools
    get_accident_records,
    get_chemical_usage,
    get_risk_assessment,
    get_regulations_data,
    get_chemical_details,
    dynamic_risk_assessment,
    # Output functions
    accident_output,
    risk_assessment_table_output,
    risk_assessment_output,
    chemical_output,
    regulations_output,
    dynamic_risk_assessment_output,
    # Utility functions
    xml_to_table,
    # Tool dictionary and LLM
    tool_dict,
    tools,
    llm_with_tools
)

# Import additional required modules
from langchain_core.messages import HumanMessage, ToolMessage

# Import Pydantic models and patch function
from pydantic_models import (
    QueryRequest,
    QueryResponse,
    RiskAssessmentTableResponse,
    ToolResponse,
    PatchRequest,
    PatchResponse,
    ResponseStatus,
    PatchResponsePosco,
    patch_response_posco,
)
from patch_url_function import patch_url_function_progress

load_dotenv()

# Configuration for patch URL and tracking
PATCH_URL = os.getenv("PATCH_URL", "https://api.example.com/status")
DEFAULT_TRACKING_ID = os.getenv("TRACKING_ID", "default-tracking-id")

# Global chat history storage (in production, use a database)
chat_history = []

# Initialize FastAPI app
app = FastAPI(
    title="DOOSAN Risk Management AI API",
    description="API for risk assessment, accident analysis, chemical safety, and compliance management",
    version="1.0.0"
)


# Pydantic models are now imported from pydantic_models.py

# API Endpoints
@app.get("/")
async def root():
    return {"message": "DOOSAN Risk Management AI API", "version": "1.0.0"}

# Individual Tool Endpoints
@app.post("/tools/accident-records", response_model=ToolResponse)
async def get_accident_records_endpoint(request: QueryRequest):
    """
    Individual endpoint for getting accident records
    """
    tracking_id = request.tracking_id or DEFAULT_TRACKING_ID
    patch_url = request.patch_url or PATCH_URL
    
    try:
        accident_docs = get_accident_records(request.query)
        output = accident_output(accident_docs=accident_docs, query=request.query)
        
        response = patch_response_posco(
            id=tracking_id, response=output, status=ResponseStatus.SUCCESS
        )
        await patch_url_function_progress(patch_url, response)
        
        return ToolResponse(
            data=output,
            success=True,
            message="Accident records retrieved successfully",
            tool_name="get_accident_records"
        )
    except Exception as e:
        response = patch_response_posco(
            id=tracking_id, response=e, status=ResponseStatus.EXCEPTION
        )
        await patch_url_function_progress(patch_url, response)
        print(f"Error retrieving accident records: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving accident records: {str(e)}")

@app.post("/tools/chemical-usage", response_model=ToolResponse)
async def get_chemical_usage_endpoint(request: QueryRequest):
    """
    Individual endpoint for getting chemical usage data
    """
    tracking_id = request.tracking_id or DEFAULT_TRACKING_ID
    patch_url = request.patch_url or PATCH_URL
    
    try:
        chemical_data = get_chemical_usage(request.query)
        # Note: You may need to add a chemical_usage_output function similar to other output functions
        
        response = patch_response_posco(
            id=tracking_id, response=chemical_data, status=ResponseStatus.SUCCESS
        )
        await patch_url_function_progress(patch_url, response)
        
        return ToolResponse(
            data=chemical_data,
            success=True,
            message="Chemical usage data retrieved successfully",
            tool_name="get_chemical_usage"
        )
    except Exception as e:
        response = patch_response_posco(
            id=tracking_id, response=e, status=ResponseStatus.EXCEPTION
        )
        await patch_url_function_progress(patch_url, response)
        print(f"Error retrieving chemical usage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chemical usage: {str(e)}")

@app.post("/tools/risk-assessment", response_model=ToolResponse)
async def get_risk_assessment_endpoint(request: QueryRequest):
    """
    Individual endpoint for getting risk assessment data
    """
    tracking_id = request.tracking_id or DEFAULT_TRACKING_ID
    patch_url = request.patch_url or PATCH_URL
    
    try:
        risk_assessment_docs = get_risk_assessment(request.query)
        output = risk_assessment_output(risk_assessment_docs=risk_assessment_docs, query=request.query)
        
        response = patch_response_posco(
            id=tracking_id, response=output, status=ResponseStatus.SUCCESS
        )
        await patch_url_function_progress(patch_url, response)
        
        return ToolResponse(
            data=output,
            success=True,
            message="Risk assessment data retrieved successfully",
            tool_name="get_risk_assessment"
        )
    except Exception as e:
        response = patch_response_posco(
            id=tracking_id, response=e, status=ResponseStatus.EXCEPTION
        )
        await patch_url_function_progress(patch_url, response)
        print(f"Error retrieving risk assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving risk assessment: {str(e)}")

@app.post("/tools/regulations", response_model=ToolResponse)
async def get_regulations_data_endpoint(request: QueryRequest):
    """
    Individual endpoint for getting regulations data
    """
    tracking_id = request.tracking_id or DEFAULT_TRACKING_ID
    patch_url = request.patch_url or PATCH_URL
    
    try:
        regulations_docs = get_regulations_data(request.query)
        output = regulations_output(regulations_docs=regulations_docs, query=request.query)
        
        response = patch_response_posco(
            id=tracking_id, response=output, status=ResponseStatus.SUCCESS
        )
        await patch_url_function_progress(patch_url, response)
        
        return ToolResponse(
            data=output,
            success=True,
            message="Regulations data retrieved successfully",
            tool_name="get_regulations_data"
        )
    except Exception as e:
        response = patch_response_posco(
            id=tracking_id, response=e, status=ResponseStatus.EXCEPTION
        )
        await patch_url_function_progress(patch_url, response)
        print(f"Error retrieving regulations data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving regulations data: {str(e)}")

@app.post("/tools/chemical-details", response_model=ToolResponse)
async def get_chemical_details_endpoint(request: QueryRequest):
    """
    Individual endpoint for getting chemical details
    """
    tracking_id = request.tracking_id or DEFAULT_TRACKING_ID
    patch_url = request.patch_url or PATCH_URL
    
    try:
        table_data = get_chemical_details(request.query)
        output = chemical_output(table_data=table_data, query=request.query)
        
        response = patch_response_posco(
            id=tracking_id, response=output, status=ResponseStatus.SUCCESS
        )
        await patch_url_function_progress(patch_url, response)
        
        return ToolResponse(
            data=output,
            success=True,
            message="Chemical details retrieved successfully",
            tool_name="get_chemical_details"
        )
    except Exception as e:
        response = patch_response_posco(
            id=tracking_id, response=e, status=ResponseStatus.EXCEPTION
        )
        await patch_url_function_progress(patch_url, response)
        print(f"Error retrieving chemical details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chemical details: {str(e)}")

@app.post("/tools/dynamic-risk-assessment", response_model=ToolResponse)
async def dynamic_risk_assessment_endpoint(request: QueryRequest):
    """
    Individual endpoint for dynamic risk assessment
    """
    tracking_id = request.tracking_id or DEFAULT_TRACKING_ID
    patch_url = request.patch_url or PATCH_URL
    
    try:
        risk_assessment_docs = dynamic_risk_assessment(request.query)
        output = dynamic_risk_assessment_output(risk_assessment_docs=risk_assessment_docs, query=request.query)
        
        response = patch_response_posco(
            id=tracking_id, response=output, status=ResponseStatus.SUCCESS
        )
        await patch_url_function_progress(patch_url, response)
        
        return ToolResponse(
            data=output,
            success=True,
            message="Dynamic risk assessment completed successfully",
            tool_name="dynamic_risk_assessment"
        )
    except Exception as e:
        response = patch_response_posco(
            id=tracking_id, response=e, status=ResponseStatus.EXCEPTION
        )
        await patch_url_function_progress(patch_url, response)
        print(f"Error performing dynamic risk assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error performing dynamic risk assessment: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint that processes user queries and returns AI responses
    """
    tracking_id = request.tracking_id or DEFAULT_TRACKING_ID
    patch_url = request.patch_url or PATCH_URL
    
    try:
        # Initialize chat history if None
        global chat_history
        if chat_history is None:
            chat_history = []
        
        # Build context with recent chat history
        context_query = request.query
        if chat_history:
            # Get the last 3 conversations for context
            recent_history = chat_history[-3:] if len(chat_history) >= 3 else chat_history
            history_context = "\n".join([f"Previous: User: {h['User']} | AI: {h['AI']}" for h in recent_history])
            context_query = f"Previous conversation context:\n{history_context}\n\nCurrent query: {request.query}"
        
        messages = [HumanMessage(context_query)]
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        output = "No response generated"
        tool_used = "none"
        
        # Process tool calls
        while ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                selected_tool = tool_dict[tool_call["name"].lower()]
                tool_name = selected_tool.name
                tool_used = tool_name
                print("Here is the selected tool name -------",tool_name)
                
                if tool_name == "get_chemical_details":
                    table_data = get_chemical_details(request.query)
                    output = chemical_output(table_data=table_data, query=request.query)
                elif tool_name == "get_regulations_data":
                    regulations_docs = get_regulations_data(request.query)
                    output = regulations_output(regulations_docs=regulations_docs, query=request.query)
                elif tool_name == "get_accident_records":
                    accident_docs = get_accident_records(request.query)
                    output = accident_output(accident_docs=accident_docs, query=request.query)
                elif tool_name == "get_risk_assessment":
                    risk_assessment_docs = get_risk_assessment(request.query)
                    output = risk_assessment_output(risk_assessment_docs=risk_assessment_docs, query=request.query)
                elif tool_name == "dynamic_risk_assessment":
                    risk_assessment_docs = dynamic_risk_assessment(request.query)
                    output = dynamic_risk_assessment_output(risk_assessment_docs=risk_assessment_docs, query=request.query)
                elif tool_name == "get_chemical_usage":
                    chemical_usage_docs = get_chemical_usage(request.query)
                    output = chemical_output(table_data=chemical_usage_docs, query=request.query)
                else:
                    output = f"Unknown tool: {tool_name}"
                
                # Add tool result to messages
                tool_message = ToolMessage(content=output, tool_call_id=tool_call["id"])
                messages.append(tool_message)
            
            # Get next AI response
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)
        
        # Get final answer
        answer = ai_msg.content if ai_msg.content else output
        
        # Add to chat history
        history = {"User": request.query, "AI": answer}
        chat_history.append(history)
        
        response = patch_response_posco(
            id=tracking_id, response=answer, status=ResponseStatus.SUCCESS
        )
        await patch_url_function_progress(patch_url, response)
        
        return QueryResponse(
            result=answer,
            tool_used=tool_used,
            success=True,
            chat_history=chat_history
        )
        
    except Exception as e:
        response = patch_response_posco(
            id=tracking_id, response=e, status=ResponseStatus.EXCEPTION
        )
        await patch_url_function_progress(patch_url, response)
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/risk-assessment-table", response_model=RiskAssessmentTableResponse)
async def risk_assessment_table_endpoint(request: QueryRequest):
    """
    Endpoint specifically for risk assessment table data
    """
    tracking_id = request.tracking_id or DEFAULT_TRACKING_ID
    patch_url = request.patch_url or PATCH_URL
    
    try:
        risk_assessment_docs = get_risk_assessment(request.query)
        json_data = risk_assessment_table_output(risk_assessment_docs=risk_assessment_docs, query=request.query)
        
        if json_data:
            # Ensure json_data is a list
            if isinstance(json_data, dict):
                json_data = [json_data]
            elif not isinstance(json_data, list):
                json_data = [json_data]
            
            response = patch_response_posco(
                id=tracking_id, response=json_data, status=ResponseStatus.SUCCESS
            )
            await patch_url_function_progress(patch_url, response)
            
            return RiskAssessmentTableResponse(
                data=json_data,
                success=True,
                message=f"Retrieved {len(json_data)} risk assessment records"
            )
        else:
            response = patch_response_posco(
                id=tracking_id, response="No data found", status=ResponseStatus.FAILED
            )
            await patch_url_function_progress(patch_url, response)
            
            return RiskAssessmentTableResponse(
                data=[],
                success=False,
                message="No valid risk assessment data to display"
            )
            
    except Exception as e:
        response = patch_response_posco(
            id=tracking_id, response=e, status=ResponseStatus.EXCEPTION
        )
        await patch_url_function_progress(patch_url, response)
        print(f"Error processing risk assessment table: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing risk assessment table: {str(e)}")

@app.get("/chat-history")
async def get_chat_history(tracking_id: Optional[str] = None, patch_url: Optional[str] = None):
    """
    Get the current chat history
    """
    tracking_id = tracking_id or DEFAULT_TRACKING_ID
    patch_url = patch_url or PATCH_URL
    
    try:
        global chat_history
        if chat_history is None:
            chat_history = []
        
        response = patch_response_posco(
            id=tracking_id, response=chat_history, status=ResponseStatus.SUCCESS
        )
        await patch_url_function_progress(patch_url, response)
        
        return {
            "chat_history": chat_history,
            "success": True,
            "message": f"Retrieved {len(chat_history)} chat messages"
        }
    except Exception as e:
        response = patch_response_posco(
            id=tracking_id, response=e, status=ResponseStatus.EXCEPTION
        )
        await patch_url_function_progress(patch_url, response)
        print(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@app.delete("/chat-history")
async def clear_chat_history(tracking_id: Optional[str] = None, patch_url: Optional[str] = None):
    """
    Clear the chat history
    """
    tracking_id = tracking_id or DEFAULT_TRACKING_ID
    patch_url = patch_url or PATCH_URL
    
    try:
        global chat_history
        chat_history = []
        
        response = patch_response_posco(
            id=tracking_id, response="Chat history cleared", status=ResponseStatus.SUCCESS
        )
        await patch_url_function_progress(patch_url, response)
        
        return {"message": "Chat history cleared successfully", "success": True}
    except Exception as e:
        response = patch_response_posco(
            id=tracking_id, response=e, status=ResponseStatus.EXCEPTION
        )
        await patch_url_function_progress(patch_url, response)
        print(f"Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
