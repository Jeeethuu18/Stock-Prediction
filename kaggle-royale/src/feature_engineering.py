import pandas as pd
import ta

def add_technical_indicators(df):
    """
    Adds technical indicators to the dataframe using the 'ta' library.
    
    Includes:
    - Daily Returns
    - SMA (10, 20)
    - RSI (14)
    - MACD
    - Bollinger Bands
    - Volume features
    
    Args:
        df (pd.DataFrame): Dataframe with 'Open', 'High', 'Low', 'Close', 'Volume'.
        
    Returns:
        pd.DataFrame: Dataframe with added features.
    """
    df = df.copy()
    
    # Ensure correct types
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    
    # 1. Daily Returns
    df['Returns'] = df['Close'].pct_change()
    
    # 2. Moving Averages
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    
    # 3. RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # 4. MACD
    # ta.trend.macd returns the MACD line (difference). 
    # We can also get signal and diff.
    df['MACD_12_26_9'] = ta.trend.macd(df['Close'])
    df['MACDs_12_26_9'] = ta.trend.macd_signal(df['Close'])
    df['MACDh_12_26_9'] = ta.trend.macd_diff(df['Close'])
    
    # 5. Bollinger Bands
    indicator_bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['BBL_20_2.0'] = indicator_bb.bollinger_lband()
    df['BBM_20_2.0'] = indicator_bb.bollinger_mavg()
    df['BBU_20_2.0'] = indicator_bb.bollinger_hband()
    
    # 6. Volume Change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    return df
