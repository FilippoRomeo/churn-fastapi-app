#!/bin/bash

echo "================================================"
echo "Starting Churn Prediction API"
echo "================================================"

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "churn-api" ]]; then
    echo "⚠️  Conda environment 'churn-api' not activated!"
    echo "Run: conda activate churn-api"
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo "✓ Python: $(which python)"
echo ""

# Run with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
