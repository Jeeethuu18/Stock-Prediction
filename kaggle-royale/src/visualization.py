import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_stock_data(df):
    """
    Creates a candlestick chart with volume.
    """
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Market Data'))
                
    fig.update_layout(
        title='Stock Price History',
        yaxis_title='Stock Price',
        xaxis_title='Date',
        template='plotly_dark'
    )
    return fig

def plot_indicators(df):
    """
    Plots SMA and Bollinger Bands.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='blue')))
    
    if 'SMA_10' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_10'], name='SMA 10', line=dict(color='orange')))
        
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='green')))
        
    fig.update_layout(title='Technical Indicators', template='plotly_dark')
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """
    Plots confusion matrix using Heatmap (Plotly).
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(cm, text_auto=True, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['DOWN', 'UP'], y=['DOWN', 'UP'],
                    title='Confusion Matrix')
    fig.update_layout(template='plotly_dark')
    return fig

def plot_feature_importance(model, feature_names):
    """
    Plots feature importance.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_imp = df_imp.sort_values(by='Importance', ascending=False)
        
        fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', title='Feature Importance')
        fig.update_layout(template='plotly_dark')
        return fig
    return None
