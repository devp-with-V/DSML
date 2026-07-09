import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from shared.registry import register_method

@register_method("random_forest_classifier")
def run_rf_classifier(X, y=None):
    if y is None:
        raise ValueError("Target y is required for classification")
        
    # Naive preprocessing for demo purposes
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    # Simple label encoding for target if it's string
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = y.astype('category').cat.codes
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "report": classification_report(y_test, preds, output_dict=True)
    }
    
    importances = dict(zip(X.columns, model.feature_importances_))
        
    return {"model": model, "metrics": metrics, "importances": importances, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
