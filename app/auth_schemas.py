"""
Pydantic schemas for authentication
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
import re

class UserCreate(BaseModel):
    """Schema for user registration"""
    email: str
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """Simple email validation"""
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError('Invalid email address')
        return v.lower()

class UserLogin(BaseModel):
    """Schema for user login"""
    username: str
    password: str

class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    """Schema for token payload"""
    username: Optional[str] = None

class UserResponse(BaseModel):
    """Schema for user data response"""
    id: int
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True
