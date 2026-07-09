import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the dataframe by inferring dtypes, reporting missing values,
    and flagging IQR-based outliers. Returns the cleaned DF and a cleaning log.
    """
    cleaning_log = {
        "initial_shape": df.shape,
        "dtypes": {},
        "missing_values": {},
        "outliers_flagged": {}
    }
    
    # Dtype inference
    df = df.convert_dtypes()
    cleaning_log["dtypes"] = df.dtypes.astype(str).to_dict()
    
    # Missing value report
    missing = df.isnull().sum()
    cleaning_log["missing_values"] = missing[missing > 0].to_dict()
    
    # IQR Outliers (only for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            cleaning_log["outliers_flagged"][col] = len(outliers)
            
    cleaning_log["final_shape"] = df.shape
    return df, cleaning_log
