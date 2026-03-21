#!/usr/bin/env python3
"""
select_model.py
Automated model selection based on test MAE performance
"""

import json
import mlflow
import os
from pathlib import Path
from mlflow.tracking import MlflowClient

# Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", 
    "https://mlflow-833456981899.us-central1.run.app/"
)

def load_model_reports():
    """Load model performance reports"""
    reports_dir = Path("reports")
    
    # Load XGBoost report
    xgb_report_path = reports_dir / "xgboost_report.json"
    if xgb_report_path.exists():
        xgb_report = json.loads(xgb_report_path.read_text())
        xgb_mae = xgb_report["test_metrics"]["mae"]
    else:
        print("❌ XGBoost report not found")
        xgb_mae = float('inf')
    
    # Load Prophet report
    prophet_report_path = reports_dir / "prophet_report.json"
    if prophet_report_path.exists():
        prophet_report = json.loads(prophet_report_path.read_text())
        prophet_mae = prophet_report["test_metrics"]["test_mae"]
    else:
        print("❌ Prophet report not found")
        prophet_mae = float('inf')
    
    return xgb_mae, prophet_mae, xgb_report if xgb_mae != float('inf') else None, prophet_report if prophet_mae != float('inf') else None

def select_best_model():
    """Select the best performing model"""
    print("🤖 Model Selection Analysis")
    print("=" * 50)
    
    xgb_mae, prophet_mae, xgb_report, prophet_report = load_model_reports()
    
    print(f"XGBoost  test MAE: {xgb_mae:.4f}")
    print(f"Prophet  test MAE: {prophet_mae:.4f}")
    
    # Determine winner
    if xgb_mae <= prophet_mae:
        best_model = "xgboost"
        best_mae = xgb_mae
        improvement = ((prophet_mae - xgb_mae) / prophet_mae * 100) if prophet_mae != float('inf') else 0
        print(f"🏆 Winner: XGBoost (MAE={best_mae:.4f}, {improvement:+.2f}% vs Prophet)")
    else:
        best_model = "prophet"
        best_mae = prophet_mae
        improvement = ((xgb_mae - prophet_mae) / xgb_mae * 100) if xgb_mae != float('inf') else 0
        print(f"🏆 Winner: Prophet (MAE={best_mae:.4f}, {improvement:+.2f}% vs XGBoost)")
    
    # Save winner
    Path("reports/best_model.txt").write_text(best_model)
    
    # Save detailed comparison
    comparison = {
        "winner": best_model,
        "winner_mae": best_mae,
        "xgboost_mae": xgb_mae,
        "prophet_mae": prophet_mae,
        "improvement_percent": improvement,
        "timestamp": mlflow.get_tracking_uri().split('//')[-1]  # Simple timestamp
    }
    
    comparison_path = Path("reports/model_comparison.json")
    comparison_path.write_text(json.dumps(comparison, indent=2))
    print(f"📊 Comparison saved to {comparison_path}")
    
    return best_model, best_mae

def promote_to_production(best_model, best_mae):
    """Promote the winning model to Production stage in MLflow"""
    print("\n🚀 Promoting to MLflow Production")
    print("=" * 50)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    model_name = f"{best_model}-supply-chain"
    
    # Get latest version of the winning model
    latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging"])
    if not latest_versions:
        print(f"❌ No versions found for {model_name}")
        return False
    
    latest_version = latest_versions[0]
    version_number = latest_version.version
    
    # Check if there's already a production model
    current_prod = client.get_latest_versions([f"{best_model}-supply-chain"], stages=["Production"])
    
    if current_prod:
        # Gate: Only promote if significantly better (5% improvement)
        current_run_id = current_prod[0].run_id
        current_mae = client.get_run(current_run_id).data.metrics.get("test_mae")
        
        if current_mae:
            improvement_threshold = 0.05  # 5% improvement required
            improvement_ratio = (current_mae - best_mae) / current_mae
            
            if improvement_ratio < improvement_threshold:
                print(f"🛑 New model not better enough for promotion")
                print(f"   Current production MAE: {current_mae:.4f}")
                print(f"   New model MAE: {best_mae:.4f}")
                print(f"   Improvement: {improvement_ratio*100:.2f}% (need {improvement_threshold*100:.1f}%+)")
                
                # Archive the new model instead of promoting
                client.transition_model_version_stage(
                    name=model_name,
                    version=version_number,
                    stage="Archived"
                )
                print(f"📦 New model archived as version {version_number}")
                return False
    
    # Promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version_number,
        stage="Production"
    )
    
    print(f"✅ Promoted {model_name} version {version_number} to Production")
    print(f"🎯 Production MAE: {best_mae:.4f}")
    
    # Archive old production models
    if current_prod:
        for old_version in current_prod:
            if old_version.version != version_number:
                client.transition_model_version_stage(
                    name=model_name,
                    version=old_version.version,
                    stage="Archived"
                )
                print(f"📦 Archived old version {old_version.version}")
    
    return True

if __name__ == "__main__":
    try:
        # Select best model
        best_model, best_mae = select_best_model()
        
        # Promote to production
        promoted = promote_to_production(best_model, best_mae)
        
        if promoted:
            print(f"\n🎉 Successfully deployed {best_model} to Production!")
        else:
            print(f"\n⏸️  {best_model} won but not promoted to Production")
        
        print(f"\n📍 Best model saved to: reports/best_model.txt")
        
    except Exception as e:
        print(f"❌ Error in model selection: {e}")
        exit(1)
