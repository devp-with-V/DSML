import pandas as pd
from shared.registry import register_method

@register_method("sarima_forecasting")
def run_sarima(X, y=None):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from sklearn.metrics import mean_squared_error
    except ImportError:
        raise ImportError("statsmodels is required for SARIMA. Run `pip install statsmodels`")
        
    series = y if y is not None else X.iloc[:, 0]
    
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    # SARIMA(1,1,1)(1,1,1,12) baseline (assuming monthly seasonality)
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
    fitted_model = model.fit(disp=False)
    
    preds = fitted_model.forecast(steps=len(test))
    
    metrics = {"mse": float(mean_squared_error(test, preds))}
    return {"model": fitted_model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
