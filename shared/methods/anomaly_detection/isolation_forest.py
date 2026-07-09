import pandas as pd
from sklearn.ensemble import IsolationForest
from shared.registry import register_method

@register_method("isolation_forest_anomaly")
def run_isolation_forest(X, y=None):
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    # Isolation Forest is tree-based and robust to varying scales
    model = IsolationForest(contamination=0.05, random_state=42)
    
    # fit_predict returns 1 for inliers, -1 for outliers/anomalies
    preds = model.fit_predict(X)
    
    anomaly_count = int(list(preds).count(-1))
    normal_count = int(list(preds).count(1))
    
    metrics = {
        "anomalies_detected": anomaly_count,
        "normal_samples": normal_count,
        "anomaly_percentage": float(anomaly_count / len(preds) * 100)
    }
    
    return {"model": model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
