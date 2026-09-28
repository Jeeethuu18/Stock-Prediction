# Stock Price Prediction System

A production-ready machine learning system to predict stock price movements (UP/DOWN) using technical indicators and daily OHLCV data.

## Features

- **Data Pipeline**: Automated cleaning, scaling, and feature engineering (RSI, MACD, SMA, Bollinger Bands).
- **Machine Learning**: Train and evaluate XGBoost, Random Forest, and Logistic Regression models.
- **Interactive UI**: Streamlit-based dashboard for easy interaction.
- **Visual Analytics**: Interactive Plotly charts for stock prices, indicators, and model performance.
- **Prediction**: Predict next day's movement with confidence scores.

## Project Structure

```
stock-price-prediction/
├── app.py                     # Main Streamlit Application
├── requirements.txt           # Python Dependencies
├── data/                      # Data Directory
├── src/                       # Source Code
│   ├── data_loader.py         # Data Ingestion
│   ├── preprocessing.py       # Data Cleaning & Scaling
│   ├── feature_engineering.py # Technical Indicators
│   ├── model_training.py      # ML Model Training
│   ├── prediction.py          # Inference Logic
│   └── visualization.py       # Plotly Charts
└── README.md                  # Project Documentation
```

## Installation

1.  **Prerequisites**: Python 3.8+ installed.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  Navigate to the project directory:
    ```bash
    cd stock-price-prediction
    ```
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  **Usage**:
    - Upload your CSV file (e.g., `yahoo_stock.csv`).
    - Go to **Data & Indicators** to inspect feature generation.
    - Go to **Model Training** to train a model (recommend XGBoost).
    - Go to **Prediction** to predict the next day's movement.

## Dataset Format

The system expects a CSV file with the following columns (headers are case-insensitive):
- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

## Models Implemented

- **XGBoost**: Gradient Boosting (Primary Recommendation)
- **Random Forest**: Ensemble Learning
- **Logistic Regression**: Baseline Linear Model

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- Confusion Matrix & Feature Importance plots.
