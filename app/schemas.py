"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

class CustomerData(BaseModel):
    """Input schema for prediction request"""
    
    # Original features
    age: int = Field(..., ge=18, le=100, description="Customer age")
    income: float = Field(..., ge=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    tenure_months: int = Field(..., ge=0, description="Months with company")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges")
    num_products: int = Field(..., ge=1, le=10, description="Number of products")
    support_calls: int = Field(..., ge=0, description="Support calls")
    complaints_last_6m: int = Field(..., ge=0, description="Recent complaints")
    avg_monthly_usage_gb: float = Field(..., ge=0, description="Monthly usage GB")
    payment_delay_days: int = Field(..., ge=0, description="Payment delays")
    
    # Engineered features
    charges_per_tenure: float = Field(..., ge=0)
    usage_per_dollar: float = Field(..., ge=0)
    complaint_rate: float = Field(..., ge=0)
    support_per_product: float = Field(..., ge=0)
    financial_stress: float
    engagement_score: float
    education_encoded: int = Field(..., ge=0, le=4)
    
    # Optional metadata
    customer_name: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = Field(None, max_length=500)
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "income": 75000.0,
                "credit_score": 720,
                "tenure_months": 24,
                "monthly_charges": 89.99,
                "num_products": 3,
                "support_calls": 2,
                "complaints_last_6m": 0,
                "avg_monthly_usage_gb": 45.5,
                "payment_delay_days": 0,
                "charges_per_tenure": 3.75,
                "usage_per_dollar": 0.506,
                "complaint_rate": 0.0,
                "support_per_product": 0.667,
                "financial_stress": -0.5,
                "engagement_score": 0.8,
                "education_encoded": 3,
                "customer_name": "John Doe",
                "notes": "High value customer"
            }
        }

class PredictionResponse(BaseModel):
    """Output schema for prediction response"""
    
    id: Optional[int] = None
    prediction: int = Field(..., description="0=No Churn, 1=Churn")
    churn_probability: float = Field(..., ge=0, le=1)
    no_churn_probability: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")
    timestamp: datetime
    customer_name: Optional[str] = None
    
    class Config:
        from_attributes = True

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    customers: List[CustomerData] = Field(..., max_items=100)

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_processed: int
    timestamp: datetime

class ErrorResponse(BaseModel):
    """Standard error response"""
    error_code: str
    error_message: str
    details: Optional[str] = None
    timestamp: datetime

class PredictionHistory(BaseModel):
    """Prediction history item"""
    id: int
    prediction: int
    churn_probability: float
    risk_level: str
    customer_name: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True
