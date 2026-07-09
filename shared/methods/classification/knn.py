import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from shared.registry import register_method

@register_method("knn_classifier")
def run_knn_classifier(X, y=None):
    if y is None: 
        raise ValueError("Target y is required for classification")
        
    X = X.fillna(X.median(numeric_only=True)).fillna("Missing")
    X = pd.get_dummies(X)
    
    if y.dtype == 'object' or y.dtype.name == 'category': 
        y = y.astype('category').cat.codes
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # KNN is distance-based, requiring scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_test_scaled)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "report": classification_report(y_test, preds, output_dict=True)
    }
    
    return {"model": model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
