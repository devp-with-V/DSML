import pandas as pd
from shared.registry import register_method

@register_method("collaborative_filtering")
def run_collab_filtering(X, y=None):
    try:
        from surprise import SVD, Dataset, Reader
        from surprise.model_selection import train_test_split
        from surprise import accuracy
    except ImportError:
        raise ImportError("Surprise library is required. Run `pip install scikit-surprise`")
        
    # Assume X is a dataframe with columns: user_id, item_id, rating
    if X.shape[1] < 3:
        raise ValueError("Collaborative filtering requires at least 3 columns: user, item, rating")
        
    reader = Reader(rating_scale=(X.iloc[:,2].min(), X.iloc[:,2].max()))
    data = Dataset.load_from_df(X.iloc[:, :3], reader)
    
    trainset, testset = train_test_split(data, test_size=0.2)
    
    model = SVD()
    model.fit(trainset)
    preds = model.test(testset)
    
    metrics = {"rmse": float(accuracy.rmse(preds, verbose=False))}
    return {"model": model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
