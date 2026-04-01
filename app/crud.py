"""
CRUD operations for database
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.models import PredictionRecord
from app.schemas import CustomerData, PredictionResponse
from typing import List, Optional

async def create_prediction(
    db: AsyncSession, 
    customer: CustomerData, 
    prediction_result: dict
) -> PredictionRecord:
    """Save prediction to database"""
    
    db_prediction = PredictionRecord(
        # Input features
        age=customer.age,
        income=customer.income,
        credit_score=customer.credit_score,
        tenure_months=customer.tenure_months,
        monthly_charges=customer.monthly_charges,
        num_products=customer.num_products,
        support_calls=customer.support_calls,
        complaints_last_6m=customer.complaints_last_6m,
        avg_monthly_usage_gb=customer.avg_monthly_usage_gb,
        payment_delay_days=customer.payment_delay_days,
        charges_per_tenure=customer.charges_per_tenure,
        usage_per_dollar=customer.usage_per_dollar,
        complaint_rate=customer.complaint_rate,
        support_per_product=customer.support_per_product,
        financial_stress=customer.financial_stress,
        engagement_score=customer.engagement_score,
        education_encoded=customer.education_encoded,
        
        # Prediction outputs
        prediction=prediction_result['prediction'],
        churn_probability=prediction_result['churn_probability'],
        no_churn_probability=prediction_result['no_churn_probability'],
        risk_level=prediction_result['risk_level'],
        
        # Metadata
        customer_name=customer.customer_name,
        notes=customer.notes
    )
    
    db.add(db_prediction)
    await db.flush()
    await db.refresh(db_prediction)
    
    return db_prediction

async def get_predictions(
    db: AsyncSession, 
    skip: int = 0, 
    limit: int = 100
) -> List[PredictionRecord]:
    """Get prediction history"""
    
    result = await db.execute(
        select(PredictionRecord)
        .order_by(desc(PredictionRecord.created_at))
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()

async def get_prediction_by_id(
    db: AsyncSession, 
    prediction_id: int
) -> Optional[PredictionRecord]:
    """Get single prediction by ID"""
    
    result = await db.execute(
        select(PredictionRecord).where(PredictionRecord.id == prediction_id)
    )
    return result.scalar_one_or_none()

async def get_statistics(db: AsyncSession) -> dict:
    """Get overall statistics"""
    
    result = await db.execute(select(PredictionRecord))
    all_predictions = result.scalars().all()
    
    if not all_predictions:
        return {
            "total_predictions": 0,
            "churn_rate": 0.0,
            "avg_churn_probability": 0.0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0
        }
    
    total = len(all_predictions)
    churned = sum(1 for p in all_predictions if p.prediction == 1)
    avg_prob = sum(p.churn_probability for p in all_predictions) / total
    
    high_risk = sum(1 for p in all_predictions if p.risk_level == "HIGH")
    medium_risk = sum(1 for p in all_predictions if p.risk_level == "MEDIUM")
    low_risk = sum(1 for p in all_predictions if p.risk_level == "LOW")
    
    return {
        "total_predictions": total,
        "churn_rate": churned / total if total > 0 else 0.0,
        "avg_churn_probability": avg_prob,
        "high_risk_count": high_risk,
        "medium_risk_count": medium_risk,
        "low_risk_count": low_risk
    }
