import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from shared.registry import register_method

@register_method("lof_anomaly")
def run_lof(X, y=None):
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    # Distance based, requires scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LOF detects outliers based on local density deviation compared to neighbors
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False)
    preds = model.fit_predict(X_scaled)
    
    anomaly_count = int(list(preds).count(-1))
    normal_count = int(list(preds).count(1))
    
    metrics = {
        "anomalies_detected": anomaly_count,
        "normal_samples": normal_count,
        "anomaly_percentage": float(anomaly_count / len(preds) * 100)
    }
    
    return {"model": model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
