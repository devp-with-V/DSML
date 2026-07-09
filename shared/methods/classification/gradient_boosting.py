import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from shared.registry import register_method

@register_method("xgboost_classifier")
def run_xgb_classifier(X, y=None):
    if y is None: 
        raise ValueError("Target y is required for classification")
        
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    if y.dtype == 'object' or y.dtype.name == 'category': 
        y = y.astype('category').cat.codes
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "report": classification_report(y_test, preds, output_dict=True)
    }
    
    importances = dict(zip(X.columns, model.feature_importances_))
    return {"model": model, "metrics": metrics, "importances": importances, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
