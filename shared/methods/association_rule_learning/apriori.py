from shared.registry import register_method

@register_method("apriori_arl")
def run_apriori(X, y=None):
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError:
        raise ImportError("mlxtend is required for Apriori. Run `pip install mlxtend`")
    
    # X must be a one-hot encoded boolean dataframe of transactions
    X_bool = X.applymap(lambda x: True if x == 1 or x == True else False)
    
    frequent_itemsets = apriori(X_bool, min_support=0.1, use_colnames=True)
    if len(frequent_itemsets) == 0:
        return {"model": None, "metrics": {"note": "No frequent itemsets found"}, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
        
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    metrics = {"num_rules": len(rules)}
    return {"model": rules, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
