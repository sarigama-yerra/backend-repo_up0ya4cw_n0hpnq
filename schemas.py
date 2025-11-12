"""
Database Schemas for Finanalyzer

Each Pydantic model represents a MongoDB collection (lowercased class name).
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class User(BaseModel):
    email: str = Field(..., description="Unique email for magic link login")
    name: Optional[str] = Field(None)
    plan: str = Field("free", description="free | paid")
    last_login_at: Optional[datetime] = None

class MagicToken(BaseModel):
    email: str
    token: str
    expires_at: datetime
    used: bool = False

class File(BaseModel):
    user_email: str
    filename: str
    content_type: str = "application/pdf"
    size_bytes: int
    storage_path: str = Field(..., description="Encrypted blob path")
    uploaded_at: datetime
    status: str = Field("uploaded", description="uploaded | processing | analyzed | failed")
    last_queried_at: Optional[datetime] = None
    fiscal_year: Optional[str] = None
    doc_type: Optional[str] = Field(None, description="balance_sheet | p_and_l | cash_flow | mixed | unknown")

class Analysis(BaseModel):
    user_email: str
    file_id: str
    health_score: float
    trends: Dict[str, Any]
    projections_1y: Dict[str, Any]
    projections_5y: Dict[str, Any]
    recommendations: List[str]
    risks: List[str]
    created_at: datetime

class ChatMessage(BaseModel):
    user_email: str
    file_id: str
    role: str = Field(..., description="user | assistant | system")
    content: str
    created_at: datetime
