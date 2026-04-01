"""
Main FastAPI application with HTMX frontend
"""
from fastapi import FastAPI, Request, Depends, HTTPException, status, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional
import logging
import joblib
import numpy as np
from pathlib import Path

from app.database import engine, Base, get_db
from app.models import PredictionRecord
from app.schemas import (
    CustomerData, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, ErrorResponse, PredictionHistory
)
from app.crud import (
    create_prediction, get_predictions, get_prediction_by_id, get_statistics
)
from app.rate_limiter import limiter, limit_predict, limit_batch, limit_general
from app.error_codes import ErrorCodes
from config import settings

# Logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model Manager
class ModelManager:
    """Singleton for model management"""
    _instance = None
    _model = None
    _feature_names = None
    _metadata = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: str):
        """Load model from pickle file"""
        try:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            logger.info(f"Loading model from {model_path}")
            artifact = joblib.load(model_path)
            
            self._model = artifact['model']
            self._feature_names = artifact['feature_names']
            self._metadata = artifact.get('metadata', {})
            
            logger.info(f"Model loaded: {len(self._feature_names)} features")
            
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> dict:
        """Make prediction"""
        if self._model is None:
            raise ValueError("Model not loaded")
        
        try:
            proba = self._model.predict_proba(features)
            pred = self._model.predict(features)
            
            return {
                'prediction': int(pred[0]),
                'churn_probability': float(proba[0][1]),
                'no_churn_probability': float(proba[0][0])
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    @property
    def feature_names(self):
        return self._feature_names
    
    @property
    def is_loaded(self):
        return self._model is not None

model_manager = ModelManager()

# Helper functions
def get_risk_level(churn_probability: float) -> str:
    """Categorize risk level"""
    if churn_probability < 0.3:
        return "LOW"
    elif churn_probability < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"

def create_error_response(error_code: str, details: str = None) -> JSONResponse:
    """Create standardized error response"""
    return JSONResponse(
        status_code=ErrorCodes.get_http_status(error_code),
        content={
            "error_code": error_code,
            "error_message": ErrorCodes.get_message(error_code),
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting application...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")
    
    # Load model
    try:
        model_manager.load_model(settings.model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="ML-powered customer churn prediction with HTMX frontend",
    version="1.0.0",
    lifespan=lifespan
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ============================================================================
# HTMX FRONTEND ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
@limit_general()
async def home(request: Request, db: AsyncSession = Depends(get_db)):
    """Main page with HTMX interface"""
    stats = await get_statistics(db)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "stats": stats}
    )

@app.get("/history", response_class=HTMLResponse)
@limit_general()
async def history_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    limit: int = 20
):
    """Prediction history page"""
    predictions = await get_predictions(db, limit=limit)
    return templates.TemplateResponse(
        "history.html",
        {"request": request, "predictions": predictions}
    )

@app.post("/htmx/predict", response_class=HTMLResponse)
@limit_predict()
async def htmx_predict(
    request: Request,
    db: AsyncSession = Depends(get_db),
    # Form fields
    customer_name: Optional[str] = Form(None),
    age: int = Form(...),
    income: float = Form(...),
    credit_score: int = Form(...),
    tenure_months: int = Form(...),
    monthly_charges: float = Form(...),
    num_products: int = Form(...),
    support_calls: int = Form(...),
    complaints_last_6m: int = Form(...),
    avg_monthly_usage_gb: float = Form(...),
    payment_delay_days: int = Form(...),
    charges_per_tenure: float = Form(...),
    usage_per_dollar: float = Form(...),
    complaint_rate: float = Form(...),
    support_per_product: float = Form(...),
    financial_stress: float = Form(...),
    engagement_score: float = Form(...),
    education_encoded: int = Form(...),
    notes: Optional[str] = Form(None)
):
    """HTMX endpoint for prediction"""
    
    if not model_manager.is_loaded:
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "error": "Model not loaded",
                "error_code": ErrorCodes.MODEL_NOT_LOADED
            }
        )
    
    try:
        # Create CustomerData object
        customer = CustomerData(
            age=age,
            income=income,
            credit_score=credit_score,
            tenure_months=tenure_months,
            monthly_charges=monthly_charges,
            num_products=num_products,
            support_calls=support_calls,
            complaints_last_6m=complaints_last_6m,
            avg_monthly_usage_gb=avg_monthly_usage_gb,
            payment_delay_days=payment_delay_days,
            charges_per_tenure=charges_per_tenure,
            usage_per_dollar=usage_per_dollar,
            complaint_rate=complaint_rate,
            support_per_product=support_per_product,
            financial_stress=financial_stress,
            engagement_score=engagement_score,
            education_encoded=education_encoded,
            customer_name=customer_name,
            notes=notes
        )
        
        # Extract features
        feature_values = [getattr(customer, f) for f in model_manager.feature_names]
        features = np.array(feature_values).reshape(1, -1)
        
        # Predict
        result = model_manager.predict(features)
        result['risk_level'] = get_risk_level(result['churn_probability'])
        
        # Save to database
        db_record = await create_prediction(db, customer, result)
        
        # Return HTMX response
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "prediction": result['prediction'],
                "churn_probability": round(result['churn_probability'], 4),
                "no_churn_probability": round(result['no_churn_probability'], 4),
                "risk_level": result['risk_level'],
                "prediction_id": db_record.id,
                "customer_name": customer.customer_name or "Unknown"
            }
        )
        
    except Exception as e:
        logger.error(f"HTMX prediction error: {e}")
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "error": str(e),
                "error_code": ErrorCodes.PREDICTION_FAILED
            }
        )

# ============================================================================
# REST API ROUTES
# ============================================================================

@app.get("/api/health")
@limit_general()
async def health_check(request: Request):
    """Health check endpoint"""
    return {
        "status": "healthy" if model_manager.is_loaded else "unhealthy",
        "model_loaded": model_manager.is_loaded,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/predict", response_model=PredictionResponse)
@limit_predict()
async def api_predict(
    customer: CustomerData,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """REST API prediction endpoint"""
    
    if not model_manager.is_loaded:
        return create_error_response(ErrorCodes.MODEL_NOT_LOADED)
    
    try:
        # Extract features
        feature_values = [getattr(customer, f) for f in model_manager.feature_names]
        features = np.array(feature_values).reshape(1, -1)
        
        # Predict
        result = model_manager.predict(features)
        result['risk_level'] = get_risk_level(result['churn_probability'])
        
        # Save to database
        db_record = await create_prediction(db, customer, result)
        
        # Return response
        return PredictionResponse(
            id=db_record.id,
            prediction=result['prediction'],
            churn_probability=round(result['churn_probability'], 4),
            no_churn_probability=round(result['no_churn_probability'], 4),
            risk_level=result['risk_level'],
            timestamp=db_record.created_at,
            customer_name=customer.customer_name
        )
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return create_error_response(ErrorCodes.PREDICTION_FAILED, str(e))

@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
@limit_batch()
async def api_batch_predict(
    batch: BatchPredictionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """REST API batch prediction endpoint"""
    
    if not model_manager.is_loaded:
        return create_error_response(ErrorCodes.MODEL_NOT_LOADED)
    
    try:
        predictions = []
        
        for customer in batch.customers:
            feature_values = [getattr(customer, f) for f in model_manager.feature_names]
            features = np.array(feature_values).reshape(1, -1)
            
            result = model_manager.predict(features)
            result['risk_level'] = get_risk_level(result['churn_probability'])
            
            db_record = await create_prediction(db, customer, result)
            
            predictions.append(PredictionResponse(
                id=db_record.id,
                prediction=result['prediction'],
                churn_probability=round(result['churn_probability'], 4),
                no_churn_probability=round(result['no_churn_probability'], 4),
                risk_level=result['risk_level'],
                timestamp=db_record.created_at,
                customer_name=customer.customer_name
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return create_error_response(ErrorCodes.PREDICTION_FAILED, str(e))

@app.get("/api/predictions", response_model=List[PredictionHistory])
@limit_general()
async def api_get_predictions(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """Get prediction history"""
    predictions = await get_predictions(db, skip, limit)
    return predictions

@app.get("/api/predictions/{prediction_id}", response_model=PredictionResponse)
@limit_general()
async def api_get_prediction(
    prediction_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Get single prediction by ID"""
    prediction = await get_prediction_by_id(db, prediction_id)
    
    if not prediction:
        return create_error_response(
            ErrorCodes.INVALID_CUSTOMER_ID,
            f"Prediction {prediction_id} not found"
        )
    
    return PredictionResponse(
        id=prediction.id,
        prediction=prediction.prediction,
        churn_probability=prediction.churn_probability,
        no_churn_probability=prediction.no_churn_probability,
        risk_level=prediction.risk_level,
        timestamp=prediction.created_at,
        customer_name=prediction.customer_name
    )

@app.get("/api/statistics")
@limit_general()
async def api_statistics(request: Request, db: AsyncSession = Depends(get_db)):
    """Get overall statistics"""
    return await get_statistics(db)

@app.get("/api/model/info")
@limit_general()
async def api_model_info(request: Request):
    """Get model information"""
    
    if not model_manager.is_loaded:
        return create_error_response(ErrorCodes.MODEL_NOT_LOADED)
    
    return {
        "model_loaded": True,
        "features": model_manager.feature_names,
        "feature_count": len(model_manager.feature_names),
        "metadata": model_manager._metadata,
        "timestamp": datetime.utcnow().isoformat()
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": ErrorCodes.INTERNAL_ERROR,
            "error_message": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error_code": ErrorCodes.INTERNAL_ERROR,
            "error_message": "An unexpected error occurred",
            "details": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
