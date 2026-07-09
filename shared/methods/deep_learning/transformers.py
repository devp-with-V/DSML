from shared.registry import register_method

@register_method("transformer_classifier")
def run_transformer(X, y=None):
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError("Transformers library required. Run `pip install transformers torch`")
        
    # Assume X is a pandas Series or list of text strings
    classifier = pipeline("sentiment-analysis")
    
    # Take a small sample to avoid massive download/compute time in a demo setting
    X_sample = list(X)[:100]
    
    # Run HuggingFace pipeline
    preds_raw = classifier(X_sample)
    
    metrics = {"note": "Transformer successfully executed on text sample", "sample_predictions": preds_raw[:5]}
    return {"model": classifier, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
