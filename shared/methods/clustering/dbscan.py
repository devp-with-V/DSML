import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from shared.registry import register_method

@register_method("dbscan_clustering")
def run_dbscan(X, y=None):
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    # Density-based, heavily relies on distance, requires scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = DBSCAN(eps=0.5, min_samples=5)
    preds = model.fit_predict(X_scaled)
    
    metrics = {}
    unique_labels = set(preds)
    # Silhouette score requires at least 2 clusters. We must ensure we have >1 cluster (excluding noise -1 if it's the only other label)
    valid_clusters = [label for label in unique_labels if label != -1]
    
    if len(valid_clusters) > 1 or (len(valid_clusters) == 1 and -1 in unique_labels):
        try:
            metrics["silhouette_score"] = float(silhouette_score(X_scaled, preds))
        except ValueError:
            metrics["silhouette_score"] = -1.0
    else:
        metrics["silhouette_score"] = -1.0
        
    metrics["noise_points"] = int(list(preds).count(-1))
        
    return {"model": model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
