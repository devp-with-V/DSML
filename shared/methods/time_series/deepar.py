from shared.registry import register_method

@register_method("deepar_forecasting")
def run_deepar(X, y=None):
    try:
        from gluonts.model.deepar import DeepAREstimator
    except ImportError:
        raise ImportError("GluonTS is required for DeepAR. Run `pip install gluonts mxnet`")
        
    metrics = {"note": "DeepAR skeleton loaded. Execution requires conversion to GluonTS ListDataset."}
    return {"model": "DeepAREstimator", "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
