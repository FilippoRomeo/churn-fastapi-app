"""
API Test Suite
"""
import requests
import json

BASE_URL = "http://localhost:8000"

SAMPLE_CUSTOMER = {
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
    "customer_name": "Test Customer"
}

def test_health():
    print("\n[TEST] Health Check")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200

def test_predict():
    print("\n[TEST] Single Prediction")
    response = requests.post(f"{BASE_URL}/api/predict", json=SAMPLE_CUSTOMER)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200

def test_batch():
    print("\n[TEST] Batch Prediction")
    response = requests.post(
        f"{BASE_URL}/api/predict/batch",
        json={"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
    )
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200

def test_history():
    print("\n[TEST] Get History")
    response = requests.get(f"{BASE_URL}/api/predictions?limit=5")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Retrieved {len(data)} predictions")
    assert response.status_code == 200

def test_statistics():
    print("\n[TEST] Get Statistics")
    response = requests.get(f"{BASE_URL}/api/statistics")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200

if __name__ == "__main__":
    print("="*70)
    print("CHURN API TEST SUITE")
    print("="*70)
    
    try:
        test_health()
        test_predict()
        test_batch()
        test_history()
        test_statistics()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
