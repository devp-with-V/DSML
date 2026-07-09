import pandas as pd
import numpy as np

def generate_eda_profile(df, target_col=None):
    """
    Generates a structured JSON dict containing EDA findings.
    """
    profile = {
        "summary": {
            "num_rows": len(df),
            "num_cols": len(df.columns)
        },
        "distributions": {},
        "correlations": {},
        "missingness": {},
        "cardinality": {}
    }
    
    # Missingness
    missing = df.isnull().mean()
    profile["missingness"] = missing[missing > 0].to_dict()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Distributions for numeric
    for col in numeric_cols:
        profile["distributions"][col] = {
            "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
            "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
            "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
            "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
            "zeros": int((df[col] == 0).sum())
        }
        
    # Cardinality for categorical
    for col in categorical_cols:
        profile["cardinality"][col] = int(df[col].nunique())
        
    # Correlations (only numeric)
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().fillna(0).to_dict()
        profile["correlations"] = corr_matrix
        
    # Target correlations if target specified
    if target_col and target_col in numeric_cols:
        profile["target_correlations"] = df[numeric_cols].corr()[target_col].fillna(0).to_dict()
        
    return profile
