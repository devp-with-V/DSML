import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from shared.registry import register_method

@register_method("pca_reduction")
def run_pca(X, y=None):
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = PCA(n_components=2)
    X_reduced = model.fit_transform(X_scaled)
    
    metrics = {
        "explained_variance_ratio": [float(v) for v in model.explained_variance_ratio_],
        "total_variance_explained": float(sum(model.explained_variance_ratio_))
    }
    
    return {"model": model, "metrics": metrics, "importances": None, "X_reduced": X_reduced, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
