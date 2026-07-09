import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from shared.registry import register_method

@register_method("linear_regression")
def run_linear_regression(X, y=None):
    if y is None:
        raise ValueError("Target y is required for regression")
        
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    metrics = {
        "mse": float(mean_squared_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds))
    }
    
    importances = dict(zip(X.columns, model.coef_))
        
    return {"model": model, "metrics": metrics, "importances": importances, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
