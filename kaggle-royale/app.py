import streamlit as st
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.data_loader import load_data
from src.feature_engineering import add_technical_indicators
from src.preprocessing import preprocess_data
from src.model_training import train_model, save_model
from src.prediction import make_prediction
from src.visualization import plot_stock_data, plot_indicators, plot_confusion_matrix, plot_feature_importance

# Page Config
st.set_page_config(page_title="Stock Price Prediction", layout="wide", page_icon="📈")

st.title("📈 Stock Price Prediction & Analysis")

# Sidebar
st.sidebar.header("DataSet Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Stock Data (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        # Load Data
        df = load_data(uploaded_file)
        st.sidebar.success("Data Loaded Successfully")
        
        # Date Range Filter
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        
        start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
        
        # Filter Data
        mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
        df_filtered = df.loc[mask]
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data & Indicators", "🧠 Model Training", "🔮 Prediction", "📉 Charts"])
        
        # --- TAB 1: Data & Indicators ---
        with tab1:
            st.subheader("Raw Data Preview")
            st.write(df_filtered.head())
            
            # Feature Engineering
            with st.spinner("Generating Technical Indicators..."):
                df_features = add_technical_indicators(df_filtered)
            
            st.subheader("Data with Technical Indicators")
            st.write(df_features.head())
            
            st.markdown("### Data Statistics")
            st.write(df_features.describe())

        # --- TAB 2: Model Training ---
        with tab2:
            st.subheader("Train Machine Learning Models")
            
            st.info("This will train XGBoost, RandomForest, and LogisticRegression to find the best performers.")
            
            if st.button("Train All Models"):
                with st.spinner("Preprocessing and Training All Models..."):
                    # Preprocess
                    processed_data = preprocess_data(df_features, scale=True)
                    X = processed_data['X']
                    y = processed_data['y']
                    scaler = processed_data['scaler']
                    feature_names = processed_data['feature_names']
                    
                    # Dictionary to store results
                    models = {}
                    metrics_results = {}
                    training_results = {}
                    
                    model_types = ["XGBoost", "RandomForest", "LogisticRegression"]
                    
                    progress_bar = st.progress(0)
                    for i, m_type in enumerate(model_types):
                        # Train
                        result = train_model(X, y, model_type=m_type)
                        models[m_type] = result['model']
                        metrics_results[m_type] = result['metrics']
                        training_results[m_type] = result
                        progress_bar.progress((i + 1) / len(model_types))
                    
                    # Save session state
                    st.session_state['models'] = models
                    st.session_state['scaler'] = scaler
                    st.session_state['result'] = training_results
                    st.session_state['feature_names'] = feature_names
                    st.session_state['df_features'] = df_features 
                    st.session_state['metrics_results'] = metrics_results
                    
                st.success("All Models Trained Successfully!")
                
                # Metrics Display
                st.subheader("Model Performance Comparison")
                metrics_df = pd.DataFrame(metrics_results).T
                st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True)
                
                # Visualization (Best Model)
                best_model_name = metrics_df['Accuracy'].idxmax()
                st.write(f"**Best Model by Accuracy:** {best_model_name}")
                
                col_chart1, col_chart2 = st.columns(2)
                best_result = training_results[best_model_name]
                with col_chart1:
                    st.write("Confusion Matrix (Best Model)")
                    st.plotly_chart(plot_confusion_matrix(best_result['y_test'], best_result['y_pred']), use_container_width=True)
                with col_chart2:
                    if best_model_name in ['XGBoost', 'RandomForest']:
                        st.write("Feature Importance (Best Model)")
                        st.plotly_chart(plot_feature_importance(models[best_model_name], feature_names), use_container_width=True)

        # --- TAB 3: Prediction ---
        with tab3:
            st.subheader("Predict Next Day Movement")
            
            if 'models' in st.session_state:
                st.info("Using ensemble of models to determine the highest confidence prediction.")
                
                # Manual Input vs Latest Data
                pred_option = st.radio("Prediction Source", ["Predict for Next Trading Day (using latest data)", "Manual Input"])
                
                if pred_option == "Predict for Next Trading Day (using latest data)":
                    latest_data = st.session_state['df_features'].iloc[[-1]] # Last row
                    
                    st.write("Latest Data Point (Date: {})".format(latest_data['Date'].values[0]))
                    
                    # Extract features
                    features_for_pred = latest_data[st.session_state['feature_names']]
                    
                    if st.button("Predict Next Day"):
                        best_model = None
                        best_conf = 0.0
                        best_pred_res = None
                        all_preds = []
                        
                        for name, model in st.session_state['models'].items():
                            res = make_prediction(model, st.session_state['scaler'], features_for_pred)
                            prob = res['probability']
                            # Confidence is distance from 0.5 (uncertainty) to 0 or 1.
                            # Scaled to 0.5-1.0 range, or just use max(p, 1-p)
                            conf = max(prob, 1 - prob)
                            
                            all_preds.append({
                                "Model": name,
                                "Prediction": res['prediction'],
                                "Confidence": f"{conf:.2%}"
                            })
                            
                            if conf > best_conf:
                                best_conf = conf
                                best_model = name
                                best_pred_res = res
                        
                        # Display Best Result
                        st.divider()
                        st.subheader(f"Selected Prediction (based on {best_model})")
                        
                        pred_label = best_pred_res['prediction']
                        color = "green" if pred_label == "UP" else "red"
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"## <span style='color:{color}'>{pred_label}</span>", unsafe_allow_html=True) 
                            st.caption("Predicted Direction")
                        with col2:
                            st.markdown(f"## {best_conf:.2%}", unsafe_allow_html=True)
                            st.caption(f"Confidence Score")
                            
                        st.info(f"The system selected **{best_model}** as it had the highest confidence score.")
                                    
                        # Show all
                        with st.expander("See details for all models"):
                            st.table(pd.DataFrame(all_preds))
                        
                else:
                    st.write("Manual feature input not fully implemented for all indicators automatically. Please use 'Latest Data' for best results.")
            else:
                st.warning("Please train models first in the 'Model Training' tab.")

        # --- TAB 4: Charts ---
        with tab4:
            st.subheader("Interactive Charts")
            
            st.plotly_chart(plot_stock_data(df_filtered), use_container_width=True)
            
            # Check if indicators exist
            if 'SMA_10' in df_features.columns:
                 st.plotly_chart(plot_indicators(df_features), use_container_width=True)

    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.error("Please ensure the CSV has correct format: Date, Open, High, Low, Close, Volume")

else:
    st.info("Please upload a CSV file to begin. \n\nTip: You can use 'yahoo_stock.csv' if you have it locally.")
    
    # Optional: Load sample button if file exists locally in predictable path
    sample_path = os.path.join("data", "sample_stock_data.csv")
    if os.path.exists(sample_path):
        if st.button("Load Sample Data"):
            # Mock upload
            pass # Too complex to mock upload object easily, user should upload
