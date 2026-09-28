import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model(X, y, model_type='XGBoost'):
    """
    Trains a model and evaluates it.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        model_type (str): 'LogisticRegression', 'RandomForest', or 'XGBoost'.
        
    Returns:
        dict: Contains 'model', 'metrics', 'X_test', 'y_test', 'y_pred'.
    """
    # Time-series aware split (no random shuffle)
    # Actually train_test_split with shuffle=False simulates a simple time-series split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    if model_type == 'LogisticRegression':
        model = LogisticRegression()
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError("Invalid model type")
        
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC AUC': roc_auc_score(y_test, y_prob)
    }
    
    return {
        'model': model,
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_trained_model(filepath):
    return joblib.load(filepath)
