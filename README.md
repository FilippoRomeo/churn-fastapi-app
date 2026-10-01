# 🔮 Churn Prediction API

Production-ready FastAPI application for predicting customer churn using Machine Learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

- ✅ **FastAPI REST API** with automatic OpenAPI docs
- ✅ **HTMX Frontend** with Tailwind CSS styling
- ✅ **SQLAlchemy Async ORM** with SQLite database
- ✅ **Rate Limiting** using SlowAPI
- ✅ **Error Handling** with standardized error codes
- ✅ **Random Forest ML Model** for churn prediction
- ✅ **Prediction History** stored in database
- ✅ **17 Engineered Features** for accurate predictions

## 🏗️ Architecture
```
FastAPI Server
├── HTMX Frontend (Tailwind CSS)
├── REST API Endpoints
├── Random Forest Model (scikit-learn)
├── SQLite Database (SQLAlchemy Async)
├── Rate Limiting (SlowAPI)
└── Error Handling (Custom Codes)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/churn-fastapi-app.git
cd churn-fastapi-app

# Create conda environment
conda create -n churn-api python=3.11 -y
conda activate churn-api

# Install dependencies
pip install -r requirements.txt
```

### Setup
```bash
# 1. Place your dataset in data/ folder
cp /path/to/cc_underwriting_5k_stratified.csv data/

# 2. Train the model
python train_model.py

# 3. Run the API
bash run.sh
```

### Access the Application

- **Web Interface**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📊 API Endpoints

### REST API

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| GET | `/api/health` | Health check | 200/min |
| POST | `/api/predict` | Single prediction | 60/min |
| POST | `/api/predict/batch` | Batch predictions | 10/min |
| GET | `/api/predictions` | Get history | 100/min |
| GET | `/api/predictions/{id}` | Get by ID | 100/min |
| GET | `/api/statistics` | Get stats | 100/min |
| GET | `/api/model/info` | Model info | 30/min |

### Example Request
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "education_encoded": 3
  }'
```

### Example Response
```json
{
  "id": 1,
  "prediction": 0,
  "churn_probability": 0.2341,
  "no_churn_probability": 0.7659,
  "risk_level": "LOW",
  "timestamp": "2026-04-01T10:30:00"
}
```

## 🎯 Features Explained

### Input Features (17)

1. **age** - Customer age (18-100)
2. **income** - Annual income
3. **credit_score** - FICO score (300-850)
4. **tenure_months** - Months with company
5. **monthly_charges** - Monthly bill
6. **num_products** - Number of products
7. **support_calls** - Support interactions
8. **complaints_last_6m** - Recent complaints
9. **avg_monthly_usage_gb** - Usage amount
10. **payment_delay_days** - Payment delays
11. **education_encoded** - Education level (0-4)

**Engineered Features:**
12. **charges_per_tenure** - Cost per month ratio
13. **usage_per_dollar** - Value ratio
14. **complaint_rate** - Complaint frequency
15. **support_per_product** - Support per product
16. **financial_stress** - Financial burden indicator
17. **engagement_score** - Customer engagement metric

## 📁 Project Structure
```
churn-fastapi-app/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── models.py            # Database models
│   ├── schemas.py           # Pydantic schemas
│   ├── crud.py              # Database operations
│   ├── database.py          # DB configuration
│   ├── rate_limiter.py      # Rate limiting
│   └── error_codes.py       # Error definitions
├── templates/
│   ├── index.html           # Main page
│   ├── predict.html         # Results template
│   └── history.html         # History page
├── static/
│   └── style.css            # Custom styles
├── saved_models/
│   └── churn_model.pkl      # Trained model
├── data/
│   └── (your dataset)
├── train_model.py           # Model training
├── test_api.py              # API tests
├── run.sh                   # Launch script
├── requirements.txt         # Dependencies
├── config.py                # Configuration
├── .env                     # Local environment variables (not committed)
├── .env.example             # Safe configuration template
└── README.md                # This file
```

## 🔧 Configuration

`.env` is ignored by Git. Copy the safe template and replace the placeholder secrets locally:

```bash
cp .env.example .env
openssl rand -hex 32
```

Use the generated value for `SECRET_KEY` in `.env`. If you use Docker Compose, also replace `POSTGRES_PASSWORD` before starting the stack.

Example local configuration:

```bash
DATABASE_URL=sqlite+aiosqlite:///./churn_predictions.db
MODEL_PATH=saved_models/churn_model.pkl
API_PORT=8000
RATE_LIMIT_PREDICT=60/minute
SECRET_KEY=replace-with-a-random-secret
```

If `SECRET_KEY` is omitted during a direct local run, the app generates a random per-process key. Set it explicitly for stable deployments, restarts, or multiple workers. `docker-compose.yml` requires both `SECRET_KEY` and `POSTGRES_PASSWORD` and binds PostgreSQL only to `127.0.0.1` on the host.

## 🧪 Testing
```bash
# Run test suite
python test_api.py
```

## 📈 Model Performance

- **Training Accuracy**: 83.43%
- **Test Accuracy**: 66.85%
- **AUC**: 0.6473
- **Model**: Random Forest (100 trees)
- **Features**: 17 (including engineered)

## 🐛 Error Codes

| Code | Type | Description |
|------|------|-------------|
| ERR_1001 | Client | Invalid input format |
| ERR_1002 | Client | Missing required field |
| ERR_1003 | Client | Validation failed |
| ERR_1004 | Client | Rate limit exceeded |
| ERR_5001 | Server | Model not loaded |
| ERR_5002 | Server | Prediction failed |
| ERR_5003 | Server | Internal error |

## 🚢 Deployment

### Docker (Optional)

Copy `.env.example` to `.env`, replace the placeholder secrets, then use Docker Compose:

```bash
cp .env.example .env
docker compose up --build
```

### Production

For production deployment, consider:
- Use PostgreSQL instead of SQLite
- Set a persistent high-entropy `SECRET_KEY`
- Use Gunicorn with multiple workers
- Set up HTTPS with nginx
- Add monitoring (Prometheus/Grafana)
- Implement CI/CD pipeline

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)

## 🙏 Acknowledgments

- FastAPI for the amazing framework
- scikit-learn for ML capabilities
- Tailwind CSS for beautiful styling
- HTMX for seamless interactivity

---

⭐ Star this repo if you find it helpful!
