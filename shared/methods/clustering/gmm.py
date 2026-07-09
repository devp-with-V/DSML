import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from shared.registry import register_method

@register_method("gmm_clustering")
def run_gmm(X, y=None):
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    # GMM is sensitive to scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GaussianMixture(n_components=3, random_state=42)
    preds = model.fit_predict(X_scaled)
    
    metrics = {}
    if len(set(preds)) > 1:
        metrics["silhouette_score"] = float(silhouette_score(X_scaled, preds))
    else:
        metrics["silhouette_score"] = -1.0
        
    # BIC and AIC are useful metrics for GMM to select number of components
    metrics["bic"] = float(model.bic(X_scaled))
    metrics["aic"] = float(model.aic(X_scaled))
        
    return {"model": model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
