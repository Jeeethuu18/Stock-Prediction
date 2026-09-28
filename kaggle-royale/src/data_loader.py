import pandas as pd
import os

def load_data(file_path):
    """
    Loads stock data from a CSV file.
    
    Args:
        file_path (str or file-like object): Path to the CSV file or a file-like object (for Streamlit).
        
    Returns:
        pd.DataFrame: Loaded dataframe with Date parsed as datetime.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Ensure column names are standard (handling generic cases)
        df.columns = [col.strip().title() for col in df.columns]
        
        # Check for required columns
        required_columns = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing columns: {missing}")
            
        # Parse Date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        raise Exception(f"Error loading data: {e}")
