import pandas as pd
from sklearn.preprocessing import StandardScaler
from shared.registry import register_method

@register_method("umap_reduction")
def run_umap(X, y=None):
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP library required. Run `pip install umap-learn`")
        
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = umap.UMAP(n_components=2, random_state=42)
    X_reduced = model.fit_transform(X_scaled)
    
    metrics = {"note": "UMAP dimensionality reduction complete"}
    
    return {"model": model, "metrics": metrics, "importances": None, "X_reduced": X_reduced, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
