import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from shared.registry import register_method

@register_method("lda_reduction")
def run_lda(X, y=None):
    if y is None: 
        raise ValueError("LDA requires a target y (it is a supervised dimensionality reduction method)")
    
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    if y.dtype == 'object' or y.dtype.name == 'category': 
        y = y.astype('category').cat.codes
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # n_components cannot be larger than min(n_features, n_classes - 1)
    # For binary classification, n_components is strictly 1
    model = LinearDiscriminantAnalysis(n_components=1) 
    X_reduced = model.fit_transform(X_scaled, y)
    
    metrics = {
        "explained_variance_ratio": [float(v) for v in model.explained_variance_ratio_]
    }
    
    return {"model": model, "metrics": metrics, "importances": None, "X_reduced": X_reduced, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
