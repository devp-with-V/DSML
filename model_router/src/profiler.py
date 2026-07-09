import pandas as pd
import numpy as np

def profile_dataset(df, target_col=None):
    """
    Computes meta-features of the dataset for the LLM router to make an informed decision.
    """
    profile = {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "target_column": target_col,
        "dtypes": df.dtypes.astype(str).value_counts().to_dict(),
        "problem_type": "clustering" if not target_col else "unknown",
        "class_balance": None,
        "cardinality": {}
    }
    
    if target_col and target_col in df.columns:
        # Determine if classification or regression
        target_series = df[target_col]
        if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 20:
            profile["problem_type"] = "regression"
        else:
            profile["problem_type"] = "classification"
            profile["class_balance"] = target_series.value_counts(normalize=True).to_dict()
            
    # Cardinality of categoricals
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        profile["cardinality"][col] = df[col].nunique()
        
    return profile
