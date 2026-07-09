import pandas as pd
import numpy as np
from shared.registry import register_method

@register_method("var_forecasting")
def run_var(X, y=None):
    try:
        from statsmodels.tsa.vector_ar.var_model import VAR
    except ImportError:
        raise ImportError("statsmodels is required for VAR. Run `pip install statsmodels`")
        
    # VAR is for multivariate time series
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    X = X.select_dtypes(include=[np.number])
    
    train_size = int(len(X) * 0.8)
    train, test = X[:train_size], X[train_size:]
    
    model = VAR(train)
    fitted_model = model.fit(maxlags=1)
    
    preds = fitted_model.forecast(fitted_model.y, steps=len(test))
    metrics = {"note": "VAR multivariate forecasting complete"}
    
    return {"model": fitted_model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
