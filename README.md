# Churn Prediction FastAPI Application

Complete ML-powered churn prediction API with SQLAlchemy database, HTMX frontend, Tailwind CSS, rate limiting, and error handling.

## 🚀 Quick Start

### 1. Create Conda Environment
```bash
cd /home/romeo/projects/code/complete-ml-masterclass/churn-fastapi-app

# Create environment
conda create -n churn-api python=3.11 -y

# Activate it
conda activate churn-api
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Place Dataset

Copy `cc_underwriting_5k_stratified.csv` to the `data/` directory:
```bash
cp ../cc_underwriting_5k_stratified.csv data/
```

### 4. Train Model
```bash
python train_model.py
```

This will:
- Load your dataset
- Engineer features
- Train Random Forest model
- Save to `saved_models/churn_model.pkl`

### 5. Run API
```bash
bash run.sh
```

Or manually:
```bash
conda activate churn-api
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Access Application

- **HTMX Frontend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🎨 UI Stack

- **Frontend**: HTMX for interactivity
- **Styling**: Tailwind CSS (CDN)
- **Backend**: FastAPI + Jinja2 templates

## 📁 Project Structure
```
churn-fastapi-app/
├── app/
│   ├── __init__.py          # Package init
│   ├── main.py              # FastAPI routes + HTMX
│   ├── models.py            # SQLAlchemy models
│   ├── schemas.py           # Pydantic schemas
│   ├── crud.py              # Database operations
│   ├── database.py          # DB configuration
│   ├── rate_limiter.py      # Rate limiting
│   └── error_codes.py       # Error definitions
├── templates/
│   ├── index.html           # Main HTMX interface (Tailwind)
│   ├── predict.html         # Prediction result (Tailwind)
│   └── history.html         # Prediction history (Tailwind)
├── static/
│   └── style.css            # Custom Tailwind styles
├── saved_models/
│   └── churn_model.pkl      # Trained model
├── data/
│   └── cc_underwriting_5k_stratified.csv
├── train_model.py           # Model training script
├── test_api.py              # API tests
├── run.sh                   # Launch script
├── requirements.txt         # Dependencies
├── config.py                # Configuration
└── .env                     # Environment variables
```

## 🎯 Features

- ✅ FastAPI REST API
- ✅ HTMX interactive frontend
- ✅ Tailwind CSS styling
- ✅ SQLAlchemy async database (SQLite)
- ✅ Rate limiting (SlowAPI)
- ✅ Error codes (ERR_XXXX)
- ✅ Prediction history
- ✅ Real-time statistics
- ✅ Random Forest model
- ✅ 17 engineered features
- ✅ Conda environment

## 🧪 Testing
```bash
# Make sure API is running first
conda activate churn-api
python test_api.py
```

## 📦 Conda Environment Management
```bash
# Activate environment
conda activate churn-api

# Deactivate
conda deactivate

# List packages
conda list

# Export environment
conda env export > environment.yml

# Remove environment (if needed)
conda env remove -n churn-api
```

---

**Created**: April 2026  
**Version**: 1.0.0  
**Python**: 3.11 (Conda)  
**UI**: Tailwind CSS + HTMX
