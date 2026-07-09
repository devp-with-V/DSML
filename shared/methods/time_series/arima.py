import pandas as pd
from shared.registry import register_method

@register_method("arima_forecasting")
def run_arima(X, y=None):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_squared_error
    except ImportError:
        raise ImportError("statsmodels is required for ARIMA. Run `pip install statsmodels`")
        
    # Assume X is a 1D sequence or y is the sequence
    series = y if y is not None else X.iloc[:, 0]
    
    # Simple train/test split for time series (no random shuffle)
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    # Fit ARIMA(1,1,1) as a default baseline
    model = ARIMA(train, order=(1,1,1))
    fitted_model = model.fit()
    
    preds = fitted_model.forecast(steps=len(test))
    
    metrics = {"mse": float(mean_squared_error(test, preds))}
    return {"model": fitted_model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
