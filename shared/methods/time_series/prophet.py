import pandas as pd
from shared.registry import register_method

@register_method("prophet_forecasting")
def run_prophet(X, y=None):
    try:
        from prophet import Prophet
        from sklearn.metrics import mean_squared_error
    except ImportError:
        raise ImportError("Prophet is required. Run `pip install prophet`")
        
    if y is None: 
        raise ValueError("Target y is required for Prophet (y values, while X should contain datetime)")
    
    df = pd.DataFrame({'ds': X.iloc[:, 0], 'y': y})
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]
    
    model = Prophet()
    model.fit(train)
    
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    preds = forecast['yhat'].iloc[-len(test):]
    
    metrics = {"mse": float(mean_squared_error(test['y'], preds))}
    return {"model": model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
