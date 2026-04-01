"""
SQLAlchemy ORM models for database tables
"""
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.sql import func
from app.database import Base

class PredictionRecord(Base):
    """Store prediction history in database"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Input features
    age = Column(Integer, nullable=False)
    income = Column(Float, nullable=False)
    credit_score = Column(Integer, nullable=False)
    tenure_months = Column(Integer, nullable=False)
    monthly_charges = Column(Float, nullable=False)
    num_products = Column(Integer, nullable=False)
    support_calls = Column(Integer, nullable=False)
    complaints_last_6m = Column(Integer, nullable=False)
    avg_monthly_usage_gb = Column(Float, nullable=False)
    payment_delay_days = Column(Integer, nullable=False)
    
    # Engineered features
    charges_per_tenure = Column(Float, nullable=False)
    usage_per_dollar = Column(Float, nullable=False)
    complaint_rate = Column(Float, nullable=False)
    support_per_product = Column(Float, nullable=False)
    financial_stress = Column(Float, nullable=False)
    engagement_score = Column(Float, nullable=False)
    education_encoded = Column(Integer, nullable=False)
    
    # Prediction outputs
    prediction = Column(Integer, nullable=False)  # 0 or 1
    churn_probability = Column(Float, nullable=False)
    no_churn_probability = Column(Float, nullable=False)
    risk_level = Column(String(10), nullable=False)  # LOW, MEDIUM, HIGH
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    customer_name = Column(String(100), nullable=True)
    notes = Column(String(500), nullable=True)
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, prediction={self.prediction}, risk={self.risk_level})>"
