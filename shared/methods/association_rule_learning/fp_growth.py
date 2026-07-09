from shared.registry import register_method

@register_method("fpgrowth_arl")
def run_fpgrowth(X, y=None):
    try:
        from mlxtend.frequent_patterns import fpgrowth, association_rules
    except ImportError:
        raise ImportError("mlxtend is required for FP-Growth. Run `pip install mlxtend`")
        
    X_bool = X.applymap(lambda x: True if x == 1 or x == True else False)
    
    frequent_itemsets = fpgrowth(X_bool, min_support=0.1, use_colnames=True)
    if len(frequent_itemsets) == 0:
        return {"model": None, "metrics": {"note": "No frequent itemsets found"}, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
        
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    metrics = {"num_rules": len(rules)}
    return {"model": rules, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
