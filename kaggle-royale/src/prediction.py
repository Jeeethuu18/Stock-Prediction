import pandas as pd

def make_prediction(model, scaler, recent_data):
    """
    Makes a prediction for the next day.
    
    Args:
        model: Trained model.
        scaler: Fitted scaler.
        recent_data (pd.DataFrame): Single row dataframe with features for the day to predict FROM.
                                    (Typically valid features derived from the latest known data)
    
    Returns:
        dict: Prediction and Probability.
    """
    # Scale if scaler exists
    if scaler:
        recent_data_scaled = scaler.transform(recent_data)
        # scaler returns numpy array, lose column names is fine for prediction usually if order matches
        # but to be safe we can wrap back to DF if needed, or just pass array if model accepts it.
        # sklearn models accept arrays.
    else:
        recent_data_scaled = recent_data
        
    prediction = model.predict(recent_data_scaled)[0]
    probability = model.predict_proba(recent_data_scaled)[0][1] if hasattr(model, "predict_proba") else 0.5
    
    return {
        'prediction': "UP" if prediction == 1 else "DOWN",
        'probability': probability
    }
