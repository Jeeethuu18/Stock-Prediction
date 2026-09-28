import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_data(df, target_col='Target', scale=True):
    """
    Cleans data, creates target variable, and scales features.
    
    Args:
        df (pd.DataFrame): Dataframe with technical indicators.
        target_col (str): Name of the target column.
        scale (bool): Whether to scale features.
        
    Returns:
        dict: Contains 'X', 'y', 'scaler', 'feature_names', 'data' (processed df).
    """
    # 1. Create Target: 1 if Close(t+1) > Close(t) else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # 2. Drop rows with NaNs (indicators + shifted target)
    df = df.dropna()
    
    if df.empty:
        raise ValueError("Dataframe is empty after dropping NaNs. Need more history.")
    
    # 3. Select Features
    # Exclude non-numeric or future-leaking columns like 'Date'
    # Retain 'Date' maybe for indexing but not for training
    feature_cols = [col for col in df.columns if col not in ['Date', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
    # We might want to keep OHLCV if we want them as features too, but indicators are usually better.
    # Let's include OHLCV as well, except Date.
    feature_cols = [col for col in df.columns if col not in ['Date', 'Target']]
    
    X = df[feature_cols]
    y = df['Target']
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    
    return {
        'X': X,
        'y': y,
        'scaler': scaler,
        'feature_names': feature_cols,
        'data': df # Cleaned data with Target
    }
