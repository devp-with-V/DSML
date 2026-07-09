import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from shared.registry import register_method

@register_method("elasticnet_regression")
def run_elasticnet_regression(X, y=None):
    if y is None: 
        raise ValueError("Target y is required for regression")
    
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ElasticNet combines L1 (Lasso) and L2 (Ridge). l1_ratio=0.5 means 50% L1, 50% L2
    model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_test_scaled)
    metrics = {
        "mse": float(mean_squared_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds))
    }
    
    importances = dict(zip(X.columns, model.coef_))
    return {"model": model, "metrics": metrics, "importances": importances, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
