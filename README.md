# Churn Prediction API

A FastAPI application for predicting customer churn with a Random Forest model, an HTMX web interface, persistence for prediction history, and containerised local deployment.

## What it includes

- FastAPI REST API with OpenAPI documentation
- HTMX frontend with Tailwind CSS
- Random Forest churn model built with scikit-learn
- Async SQLAlchemy persistence
- Prediction history and summary statistics
- Rate limiting and structured error handling
- Docker / Docker Compose support
- Environment-based secret and database configuration

## Architecture

```text
FastAPI
├── HTMX frontend
├── REST API
├── Random Forest model
├── SQLAlchemy persistence
├── rate limiting
└── Docker deployment
```

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/FilippoRomeo/churn-fastapi-app.git
cd churn-fastapi-app
```

### 2. Create an environment

```bash
conda create -n churn-api python=3.11 -y
conda activate churn-api
pip install -r requirements.txt
```

### 3. Configure local settings

```bash
cp .env.example .env
openssl rand -hex 32
```

Use the generated value for `SECRET_KEY` in `.env`.

Example local configuration:

```ini
DATABASE_URL=sqlite+aiosqlite:///./churn_predictions.db
MODEL_PATH=saved_models/churn_model.pkl
API_PORT=8000
RATE_LIMIT_PREDICT=60/minute
SECRET_KEY=replace-with-a-random-secret
```

If you use Docker Compose, also set a strong `POSTGRES_PASSWORD`.

### 4. Train the model

Place the source dataset in `data/`, then run:

```bash
python train_model.py
```

### 5. Start the application

```bash
bash run.sh
```

Then open:

- Web UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Single prediction |
| POST | `/api/predict/batch` | Batch prediction |
| GET | `/api/predictions` | Prediction history |
| GET | `/api/predictions/{id}` | Prediction by ID |
| GET | `/api/statistics` | Aggregate statistics |
| GET | `/api/model/info` | Model information |

Example request:

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

## Model

The project uses 17 input and engineered features with a 100-tree Random Forest classifier.

Metrics recorded in the project:

- Training accuracy: 83.43%
- Test accuracy: 66.85%
- AUC: 0.6473

These metrics describe the included experiment and should not be treated as production performance guarantees.

## Docker

```bash
cp .env.example .env
# Set SECRET_KEY and POSTGRES_PASSWORD in .env
docker compose up --build
```

The Compose configuration requires explicit secrets and binds PostgreSQL to `127.0.0.1` on the host.

## Security and configuration

- Real secrets belong in `.env`, which is ignored by Git.
- `.env.example` contains placeholders only.
- JWT operations use the application settings rather than hard-coded secrets.
- A direct local run generates a random per-process fallback secret if `SECRET_KEY` is absent.
- Stable deployments and multi-worker setups should always provide an explicit high-entropy `SECRET_KEY`.

## Testing

```bash
python test_api.py
```

## Main stack

`Python` `FastAPI` `scikit-learn` `SQLAlchemy` `HTMX` `Tailwind CSS` `Docker` `PostgreSQL` `SQLite`

## License

MIT
