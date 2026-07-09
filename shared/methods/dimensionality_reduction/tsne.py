import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from shared.registry import register_method

@register_method("tsne_reduction")
def run_tsne(X, y=None):
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = TSNE(n_components=2, random_state=42)
    X_reduced = model.fit_transform(X_scaled)
    
    # t-SNE provides Kullback-Leibler divergence
    metrics = {
        "kl_divergence": float(model.kl_divergence_)
    }
    
    return {"model": model, "metrics": metrics, "importances": None, "X_reduced": X_reduced, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
