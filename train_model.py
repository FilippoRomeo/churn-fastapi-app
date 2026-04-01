"""
Train Random Forest model on cc_underwriting_5k_stratified dataset
Maps credit underwriting features to churn prediction features
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature names that the API expects
FEATURE_NAMES = [
    'age', 'income', 'credit_score', 'tenure_months', 'monthly_charges',
    'num_products', 'support_calls', 'complaints_last_6m', 
    'avg_monthly_usage_gb', 'payment_delay_days',
    'charges_per_tenure', 'usage_per_dollar', 'complaint_rate',
    'support_per_product', 'financial_stress', 'engagement_score',
    'education_encoded',
]

def load_and_prepare_data():
    """Load cc_underwriting_5k_stratified dataset"""
    logger.info("Loading dataset...")
    
    # Try to find the dataset
    possible_paths = [
        'data/cc_underwriting_5k_stratified.csv',
        '../data/cc_underwriting_5k_stratified.csv',
        'cc_underwriting_5k_stratified.csv',
    ]
    
    df = None
    for path in possible_paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            logger.info(f"Dataset loaded from {path}")
            break
    
    if df is None:
        raise FileNotFoundError(
            "Dataset not found. Please place cc_underwriting_5k_stratified.csv in "
            "the data/ directory or current directory"
        )
    
    logger.info(f"Dataset shape: {df.shape}")
    return df

def map_underwriting_to_churn_features(df):
    """
    Map credit underwriting dataset columns to churn prediction features
    """
    logger.info("Mapping underwriting features to churn features...")
    
    # Create a new dataframe with mapped features
    mapped_df = pd.DataFrame()
    
    # Direct mappings
    mapped_df['age'] = df['age']
    mapped_df['income'] = df['annual_income']
    mapped_df['credit_score'] = df['fico_score']
    
    # Derived/synthetic mappings
    # tenure_months: use credit_history_length_months or bank_relationship_years * 12
    if 'credit_history_length_months' in df.columns:
        mapped_df['tenure_months'] = df['credit_history_length_months']
    elif 'bank_relationship_years' in df.columns:
        mapped_df['tenure_months'] = df['bank_relationship_years'] * 12
    else:
        mapped_df['tenure_months'] = np.random.randint(1, 120, len(df))
    
    # monthly_charges: use total_monthly_expenses or monthly_rent_mortgage + monthly_car_payment
    if 'total_monthly_expenses' in df.columns:
        mapped_df['monthly_charges'] = df['total_monthly_expenses'] / 10  # Scale down
    else:
        mapped_df['monthly_charges'] = (
            df['monthly_rent_mortgage'].fillna(0) + 
            df['monthly_car_payment'].fillna(0)
        )
    
    # num_products: use num_open_accounts or banking_product_count
    if 'num_open_accounts' in df.columns:
        mapped_df['num_products'] = df['num_open_accounts'].clip(1, 10)
    elif 'banking_product_count' in df.columns:
        mapped_df['num_products'] = df['banking_product_count'].clip(1, 10)
    else:
        mapped_df['num_products'] = 3
    
    # support_calls: use hard_inquiries or nsf_incidents
    if 'hard_inquiries_last_12mo' in df.columns:
        mapped_df['support_calls'] = df['hard_inquiries_last_12mo']
    elif 'nsf_incidents_last_12mo' in df.columns:
        mapped_df['support_calls'] = df['nsf_incidents_last_12mo']
    else:
        mapped_df['support_calls'] = 0
    
    # complaints_last_6m: use late_payments_last_12mo or derogatory_marks_count
    if 'late_payments_last_12mo' in df.columns:
        mapped_df['complaints_last_6m'] = df['late_payments_last_12mo']
    elif 'derogatory_marks_count' in df.columns:
        mapped_df['complaints_last_6m'] = df['derogatory_marks_count']
    else:
        mapped_df['complaints_last_6m'] = 0
    
    # avg_monthly_usage_gb: use avg_transaction_amount or estimated_annual_card_spend
    if 'avg_transaction_amount' in df.columns:
        mapped_df['avg_monthly_usage_gb'] = df['avg_transaction_amount'] / 10
    elif 'estimated_annual_card_spend' in df.columns:
        mapped_df['avg_monthly_usage_gb'] = df['estimated_annual_card_spend'] / 120
    else:
        mapped_df['avg_monthly_usage_gb'] = 50.0
    
    # payment_delay_days: use max_delinquency_days_24mo
    if 'max_delinquency_days_24mo' in df.columns:
        mapped_df['payment_delay_days'] = df['max_delinquency_days_24mo'].fillna(0)
    elif 'max_delinquency_days_ever' in df.columns:
        mapped_df['payment_delay_days'] = df['max_delinquency_days_ever'].fillna(0)
    else:
        mapped_df['payment_delay_days'] = 0
    
    # Education encoding
    if 'education_level' in df.columns:
        education_map = {
            'High School Diploma': 0,
            'Some College': 1,
            'Bachelor Degree': 2,
            'Master Degree': 3,
            'PhD': 4,
            'Graduate Degree': 3,
            'Associate Degree': 1
        }
        mapped_df['education_encoded'] = df['education_level'].map(education_map).fillna(2)
    else:
        mapped_df['education_encoded'] = 2
    
    logger.info("✓ Feature mapping complete")
    return mapped_df

def engineer_features(df):
    """Create engineered features from mapped base features"""
    logger.info("Engineering features...")
    
    # Ensure numeric types
    for col in df.columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Create engineered features
    df['charges_per_tenure'] = df['monthly_charges'] / (df['tenure_months'] + 1)
    df['usage_per_dollar'] = df['avg_monthly_usage_gb'] / (df['monthly_charges'] + 1)
    df['complaint_rate'] = df['complaints_last_6m'] / (df['tenure_months'] + 1)
    df['support_per_product'] = df['support_calls'] / (df['num_products'] + 0.1)
    df['financial_stress'] = (df['monthly_charges'] / (df['income'] / 12 + 1)) - 1
    df['engagement_score'] = (df['avg_monthly_usage_gb'] / 100) * (df['num_products'] / 5)
    
    # Clip extreme values
    df['charges_per_tenure'] = df['charges_per_tenure'].clip(0, 100)
    df['usage_per_dollar'] = df['usage_per_dollar'].clip(0, 10)
    df['complaint_rate'] = df['complaint_rate'].clip(0, 5)
    df['support_per_product'] = df['support_per_product'].clip(0, 20)
    df['financial_stress'] = df['financial_stress'].clip(-5, 10)
    df['engagement_score'] = df['engagement_score'].clip(0, 10)
    
    logger.info("✓ Features engineered successfully")
    return df

def create_target(df, original_df):
    """
    Create churn target based on risk factors
    This creates a realistic churn scenario with ~25% churn rate
    """
    logger.info("Creating synthetic churn target from risk factors...")
    
    np.random.seed(42)  # For reproducibility
    
    # Calculate risk score based on multiple factors
    risk_score = np.zeros(len(df))
    
    # High complaints = high churn risk
    risk_score += (df['complaints_last_6m'] > 1).astype(float) * 0.25
    
    # Many support calls = high churn risk
    risk_score += (df['support_calls'] > 3).astype(float) * 0.20
    
    # Payment delays = high churn risk
    risk_score += (df['payment_delay_days'] > 10).astype(float) * 0.25
    
    # Low tenure = higher churn risk
    risk_score += (df['tenure_months'] < 24).astype(float) * 0.15
    
    # High financial stress = higher churn risk
    risk_score += (df['financial_stress'] > 0.3).astype(float) * 0.15
    
    # Low credit score = higher churn risk
    risk_score += (df['credit_score'] < 600).astype(float) * 0.15
    
    # Low engagement = higher churn risk
    risk_score += (df['engagement_score'] < 0.5).astype(float) * 0.10
    
    # Normalize to probability
    risk_score = risk_score / risk_score.max()
    
    # Add some randomness
    noise = np.random.uniform(-0.1, 0.1, len(df))
    churn_probability = np.clip(risk_score + noise, 0, 1)
    
    # Generate binary target
    y = (np.random.random(len(df)) < churn_probability).astype(int)
    
    churn_rate = y.mean()
    logger.info(f"✓ Synthetic churn target created with {churn_rate:.2%} churn rate")
    
    if churn_rate < 0.05:
        logger.warning("Churn rate is very low! Adjusting...")
        # Force at least 20% churn
        n_churned_needed = int(len(df) * 0.20)
        n_churned_current = y.sum()
        if n_churned_current < n_churned_needed:
            # Find customers with highest risk who aren't churned yet
            additional_churns_needed = n_churned_needed - n_churned_current
            non_churned_idx = np.where(y == 0)[0]
            high_risk_idx = non_churned_idx[np.argsort(-churn_probability[non_churned_idx])[:additional_churns_needed]]
            y[high_risk_idx] = 1
            logger.info(f"✓ Adjusted to {y.mean():.2%} churn rate")
    
    return y

def train_model(X_train, y_train):
    """Train Random Forest model"""
    logger.info("Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    logger.info("✓ Model training complete")
    
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance"""
    logger.info("Evaluating model...")
    
    # Check if model can predict probabilities
    if not hasattr(model, 'predict_proba'):
        raise ValueError("Model doesn't support probability predictions")
    
    # Training metrics
    train_pred = model.predict(X_train)
    train_proba = model.predict_proba(X_train)
    
    # Check if we have both classes
    if train_proba.shape[1] < 2:
        raise ValueError(f"Model only predicts {train_proba.shape[1]} class. Need both classes.")
    
    train_proba_pos = train_proba[:, 1]
    train_score = model.score(X_train, y_train)
    train_auc = roc_auc_score(y_train, train_proba_pos)
    
    # Test metrics
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_score = model.score(X_test, y_test)
    test_auc = roc_auc_score(y_test, test_proba)
    
    logger.info(f"Training Accuracy: {train_score:.4f}")
    logger.info(f"Training AUC: {train_auc:.4f}")
    logger.info(f"Test Accuracy: {test_score:.4f}")
    logger.info(f"Test AUC: {test_auc:.4f}")
    
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT (Test Set)")
    print("="*70)
    print(classification_report(y_test, test_pred, zero_division=0))
    
    print("\nCONFUSION MATRIX (Test Set)")
    print("="*70)
    cm = confusion_matrix(y_test, test_pred)
    print(f"True Negatives:  {cm[0][0]:4d} | False Positives: {cm[0][1]:4d}")
    print(f"False Negatives: {cm[1][0]:4d} | True Positives:  {cm[1][1]:4d}")
    
    return {
        'train_accuracy': float(train_score),
        'train_auc': float(train_auc),
        'test_accuracy': float(test_score),
        'test_auc': float(test_auc)
    }

def save_model(model, feature_names, metrics):
    """Save model to pickle file"""
    logger.info("Saving model...")
    
    Path('saved_models').mkdir(exist_ok=True)
    
    model_artifact = {
        'model': model,
        'feature_names': feature_names,
        'metadata': {
            'model_type': 'RandomForestClassifier',
            'n_features': len(feature_names),
            'train_accuracy': metrics['train_accuracy'],
            'train_auc': metrics['train_auc'],
            'test_accuracy': metrics['test_accuracy'],
            'test_auc': metrics['test_auc'],
            'created_at': datetime.now().isoformat(),
            'sklearn_version': '1.4.0',
            'dataset': 'cc_underwriting_5k_stratified (mapped to churn features)'
        }
    }
    
    model_path = 'saved_models/churn_model.pkl'
    joblib.dump(model_artifact, model_path, compress=3)
    
    logger.info(f"✓ Model saved to {model_path}")
    
    # Print feature importance
    print("\n" + "="*70)
    print("TOP 10 FEATURE IMPORTANCE")
    print("="*70)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices, 1):
        print(f"{i:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")
    
    return model_path

def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("CHURN MODEL TRAINING PIPELINE")
    print("Credit Underwriting Dataset → Churn Features")
    print("="*70 + "\n")
    
    try:
        # Load original data
        original_df = load_and_prepare_data()
        
        # Map to churn features
        df = map_underwriting_to_churn_features(original_df)
        
        # Engineer features
        df = engineer_features(df)
        
        # Create target
        y = create_target(df, original_df)
        
        # Prepare features
        X = df[FEATURE_NAMES].values
        
        # Check for NaN/Inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("Found NaN/Inf values, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        logger.info(f"Train churn rate: {y_train.mean():.2%}")
        logger.info(f"Test churn rate: {y_test.mean():.2%}")
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        # Save model
        model_path = save_model(model, FEATURE_NAMES, metrics)
        
        print("\n" + "="*70)
        print("✓ MODEL TRAINING COMPLETE")
        print("="*70)
        print(f"\nModel saved to: {model_path}")
        print("\nFeature Mapping:")
        print("  age              ← age")
        print("  income           ← annual_income")
        print("  credit_score     ← fico_score")
        print("  tenure_months    ← credit_history_length_months")
        print("  monthly_charges  ← total_monthly_expenses / 10")
        print("  num_products     ← num_open_accounts")
        print("  support_calls    ← hard_inquiries_last_12mo")
        print("  complaints_last_6m ← late_payments_last_12mo")
        print("  avg_monthly_usage_gb ← avg_transaction_amount / 10")
        print("  payment_delay_days ← max_delinquency_days_24mo")
        print("  education_encoded ← education_level (mapped)")
        print("\nNext steps:")
        print("1. Review the metrics above")
        print("2. Run the API: bash run.sh")
        print("3. Test at: http://localhost:8000")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
