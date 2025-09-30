from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from enum import Enum


class ResponseStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    EXCEPTION = "EXCEPTION"
    PROCESSING = "PROCESSING"


class PatchResponsePosco(BaseModel):
    id: str
    response: Any
    status: ResponseStatus


class QueryRequest(BaseModel):
    query: str
    tracking_id: Optional[str] = None
    patch_url: Optional[str] = None


class QueryResponse(BaseModel):
    result: str
    tool_used: str
    success: bool
    chat_history: List[Dict[str, str]]


class RiskAssessmentTableResponse(BaseModel):
    data: List[Dict[str, Any]]
    success: bool
    message: str


class ToolResponse(BaseModel):
    data: Any
    success: bool
    message: str
    tool_name: str


class PatchRequest(BaseModel):
    url: str
    data: Dict[str, Any]
    uuid: Optional[str] = None


class PatchResponse(BaseModel):
    success: bool
    message: str
    response_data: Optional[Dict[str, Any]] = None


class TrackingRequest(BaseModel):
    tracking_id: Optional[str] = None
    patch_url: Optional[str] = None


def patch_response_posco(id: str, response: Any, status: ResponseStatus) -> PatchResponsePosco:
    """Helper function to create patch response for POSCO"""
    return PatchResponsePosco(id=id, response=response, status=status)
