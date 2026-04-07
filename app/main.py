"""
Main FastAPI application with HTMX frontend and Authentication
"""
from fastapi import FastAPI, Request, Depends, HTTPException, status, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional
import logging
import joblib
import numpy as np
from pathlib import Path

from app.database import engine, Base, get_db
from app.models import PredictionRecord
from app.auth_models import User
from app.schemas import (
    CustomerData, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, ErrorResponse, PredictionHistory
)
from app.auth_schemas import UserCreate, UserLogin, Token, UserResponse
from app.crud import (
    create_prediction, get_predictions, get_prediction_by_id, get_statistics
)
from app.auth import (
    get_password_hash, authenticate_user, create_access_token,
    get_current_active_user, get_user_by_username, get_user_by_email,
    ACCESS_TOKEN_EXPIRE_MINUTES
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
    description="ML-powered customer churn prediction with authentication",
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
# AUTHENTICATION ROUTES
# ============================================================================

@app.post("/api/auth/signup", response_model=UserResponse)
@limit_general()
async def signup(user_data: UserCreate, request: Request, db: AsyncSession = Depends(get_db)):
    """Register a new user"""
    
    # Check if user already exists
    existing_user = await get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    existing_username = await get_user_by_username(db, user_data.username)
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    logger.info(f"New user registered: {new_user.username}")
    
    return new_user

@app.post("/api/auth/login", response_model=Token)
@limit_general()
async def login(user_credentials: UserLogin, request: Request, db: AsyncSession = Depends(get_db)):
    """Login and get access token"""
    
    user = await authenticate_user(db, user_credentials.username, user_credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {user.username}")
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user

# ============================================================================
# HTMX FRONTEND ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
@limit_general()
async def home(request: Request):
    """Login page"""
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
@limit_general()
async def signup_page(request: Request):
    """Signup page"""
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
@limit_general()
async def dashboard(request: Request, db: AsyncSession = Depends(get_db)):
    """Dashboard page"""
    stats = await get_statistics(db)
    return templates.TemplateResponse("dashboard.html", {"request": request, "stats": stats})

@app.get("/history", response_class=HTMLResponse)
@limit_general()
async def history_page(request: Request, db: AsyncSession = Depends(get_db), limit: int = 20):
    """Prediction history page"""
    predictions = await get_predictions(db, limit=limit)
    return templates.TemplateResponse("history.html", {"request": request, "predictions": predictions})

@app.post("/htmx/predict", response_class=HTMLResponse)
@limit_predict()
async def htmx_predict(
    request: Request,
    db: AsyncSession = Depends(get_db),
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
        customer = CustomerData(
            age=age, income=income, credit_score=credit_score,
            tenure_months=tenure_months, monthly_charges=monthly_charges,
            num_products=num_products, support_calls=support_calls,
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
        
        feature_values = [getattr(customer, f) for f in model_manager.feature_names]
        features = np.array(feature_values).reshape(1, -1)
        
        result = model_manager.predict(features)
        result['risk_level'] = get_risk_level(result['churn_probability'])
        
        db_record = await create_prediction(db, customer, result)
        
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
    """Health check endpoint (public)"""
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """REST API prediction endpoint (requires authentication)"""
    
    if not model_manager.is_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        feature_values = [getattr(customer, f) for f in model_manager.feature_names]
        features = np.array(feature_values).reshape(1, -1)
        
        result = model_manager.predict(features)
        result['risk_level'] = get_risk_level(result['churn_probability'])
        
        db_record = await create_prediction(db, customer, result)
        
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions", response_model=List[PredictionHistory])
@limit_general()
async def api_get_predictions(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get prediction history (requires authentication)"""
    predictions = await get_predictions(db, skip, limit)
    return predictions

@app.get("/api/statistics")
@limit_general()
async def api_statistics(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get overall statistics (requires authentication)"""
    return await get_statistics(db)
