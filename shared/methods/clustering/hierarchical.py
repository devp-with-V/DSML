import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from shared.registry import register_method

@register_method("hierarchical_clustering")
def run_hierarchical(X, y=None):
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    # Distance based, requires scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Agglomerative clustering (bottom-up hierarchical)
    model = AgglomerativeClustering(n_clusters=3)
    preds = model.fit_predict(X_scaled)
    
    metrics = {}
    if len(set(preds)) > 1:
        metrics["silhouette_score"] = float(silhouette_score(X_scaled, preds))
    else:
        metrics["silhouette_score"] = -1.0
        
    return {"model": model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
