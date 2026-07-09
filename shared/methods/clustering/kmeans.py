import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from shared.registry import register_method

@register_method("kmeans_clustering")
def run_kmeans(X, y=None):
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    preds = model.fit_predict(X)
    
    metrics = {}
    if len(set(preds)) > 1:
        metrics["silhouette_score"] = float(silhouette_score(X, preds))
    else:
        metrics["silhouette_score"] = -1.0
        
    return {"model": model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
