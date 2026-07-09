import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from shared.registry import register_method

@register_method("logistic_regression")
def run_logistic_regression(X, y=None):
    if y is None:
        raise ValueError("Target y is required for classification")
        
    # Naive preprocessing
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    # Simple label encoding for target if it's string
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = y.astype('category').cat.codes
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Logistic Regression is sensitive to feature scaling, so we standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_test_scaled)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "report": classification_report(y_test, preds, output_dict=True)
    }
    
    importances = None
    if hasattr(model, "coef_"):
        # For multi-class, coef_ shape is (n_classes, n_features). 
        # We take the mean absolute importance across all classes to gauge overall feature importance.
        avg_coefs = np.mean(np.abs(model.coef_), axis=0)
        importances = dict(zip(X.columns, avg_coefs))
        
    return {"model": model, "metrics": metrics, "importances": importances, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
