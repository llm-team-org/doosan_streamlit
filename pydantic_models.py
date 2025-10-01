# pylint: disable=all
# fmt: off
# flake8: noqaimport os
from typing import Optional, List
from pydantic import BaseModel

class UnifiedQueryRequest(BaseModel):
    request_id: int
    query: str
    type: Optional[int] =None

class UnifiedQueryResponse(BaseModel):
    request_id: int
    type: Optional[int] =None
    status: int
    answer: Optional[str] = None
    sources: Optional[List] = None