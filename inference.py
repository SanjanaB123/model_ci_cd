#!/usr/bin/env python3
"""
inference.py
Model-agnostic inference using MLflow Production models
"""

import mlflow
import os
import pandas as pd
import json
from pathlib import Path

# Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", 
    "https://mlflow-833456981899.us-central1.run.app/"
)

def load_production_model():
    """Load the current production model from MLflow"""
    print("🔄 Loading Production Model")
    print("=" * 40)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Try XGBoost first, then Prophet
    models_to_try = [
        "xgboost-supply-chain",
        "prophet-supply-chain"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"📦 Attempting to load: {model_name}/Production")
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
            print(f"✅ Successfully loaded {model_name} from Production")
            return model, model_name
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            continue
    
    raise Exception("No production model found in MLflow")

def predict_demand(model, input_data, model_name):
    """Make predictions using the loaded model"""
    print(f"\n🔮 Making Predictions with {model_name}")
    print("=" * 40)
    
    # Ensure input_data is a DataFrame
    if isinstance(input_data, (dict, list)):
        input_df = pd.DataFrame(input_data)
    else:
        input_df = input_data.copy()
    
    print(f"📊 Input shape: {input_df.shape}")
    print(f"📊 Input columns: {list(input_df.columns)}")
    
    # Make predictions
    predictions = model.predict(input_df)
    
    # Ensure non-negative predictions
    if hasattr(predictions, 'clip'):
        predictions = predictions.clip(lower=0)
    elif isinstance(predictions, list):
        predictions = [max(0, p) for p in predictions]
    
    print(f"🎯 Predictions shape: {len(predictions)}")
    print(f"📈 Sample predictions: {predictions[:5] if len(predictions) > 5 else predictions}")
    
    return predictions

def get_model_info(model_name):
    """Get information about the production model"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Get latest production version
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            version = prod_versions[0]
            run_id = version.run_id
            
            # Get run info
            run = client.get_run(run_id)
            metrics = run.data.metrics
            params = run.data.params
            
            info = {
                "model_name": model_name,
                "version": version.version,
                "stage": version.stage,
                "run_id": run_id,
                "test_mae": metrics.get("test_mae"),
                "test_rmse": metrics.get("test_rmse"),
                "test_r2": metrics.get("test_r2"),
                "model_type": params.get("model_type"),
                "pipeline_version": params.get("pipeline_version"),
                "creation_time": run.info.start_time
            }
            
            return info
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
    
    return None

def main():
    """Main inference function"""
    try:
        # Load production model
        model, model_name = load_production_model()
        
        # Get model information
        model_info = get_model_info(model_name)
        if model_info:
            print(f"\n📋 Model Information:")
            print(f"   Name: {model_info['model_name']}")
            print(f"   Version: {model_info['version']}")
            print(f"   Type: {model_info['model_type']}")
            print(f"   Test MAE: {model_info['test_mae']:.4f}")
            print(f"   Test R²: {model_info['test_r2']:.4f}")
        
        # Example usage with sample data
        print(f"\n🧪 Running Sample Inference")
        print("=" * 40)
        
        # Create sample input (adjust columns based on your feature set)
        sample_data = {
            'sales_lag_7': [100, 150, 120],
            'sales_lag_14': [95, 145, 115],
            'sales_lag_28': [90, 140, 110],
            'sales_roll_mean_7': [98, 148, 118],
            'sales_roll_mean_14': [96, 146, 116],
            'sales_roll_mean_28': [92, 142, 112],
            'dow': [1, 2, 3],  # Day of week
            'month': [1, 1, 1],  # Month
            'is_weekend': [0, 0, 0],
            'Holiday/Promotion': [0, 1, 0],
            'Discount': [0.1, 0.15, 0.05],
            'Lead Time Days': [7, 5, 10],
            'Category_enc': [1, 2, 1],
            'Region_enc': [1, 1, 2],
            'Seasonality_enc': [1, 1, 1],
            'series_enc': [1, 1, 1]
        }
        
        # Make predictions
        predictions = predict_demand(model, sample_data, model_name)
        
        print(f"\n🎉 Inference Complete!")
        print(f"   Model: {model_name}")
        print(f"   Predictions: {predictions}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        raise

if __name__ == "__main__":
    main()
