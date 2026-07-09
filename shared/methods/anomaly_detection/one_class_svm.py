import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from shared.registry import register_method

@register_method("one_class_svm_anomaly")
def run_one_class_svm(X, y=None):
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    # Distance/Kernel based, highly dependent on scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # nu represents an upper bound on the fraction of training errors (anomalies)
    model = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    preds = model.fit_predict(X_scaled)
    
    anomaly_count = int(list(preds).count(-1))
    normal_count = int(list(preds).count(1))
    
    metrics = {
        "anomalies_detected": anomaly_count,
        "normal_samples": normal_count,
        "anomaly_percentage": float(anomaly_count / len(preds) * 100)
    }
    
    return {"model": model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
