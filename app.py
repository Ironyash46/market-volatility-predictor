import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# 1. APP CONFIGURATION
st.set_page_config(page_title="Stock Volatility Predictor", layout="wide")
st.title("📈 Stock Volatility Predictor")
st.markdown("Enter a stock ticker to train a live model and predict volatility.")

# 2. SIDEBAR INPUTS
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
years = st.sidebar.slider("Years of Data", 1, 10, 5)

# 3. HELPER FUNCTIONS
@st.cache_data
def load_data(ticker, years):
    """Fetches data from Yahoo Finance"""
    start_date = pd.Timestamp.now() - pd.DateOffset(years=years)
    df = yf.download(ticker, start=start_date, end=pd.Timestamp.now())
    if df.empty:
        return None
    return df

def prepare_data(df):
    """Feature Engineering"""
    data = df.copy()
    # Calculate Returns and Volatility (21-day rolling std dev)
    data['Returns'] = data['Close'].pct_change()
    data['Volatility_21d'] = data['Returns'].rolling(window=21).std()
    
    # Features
    data['Lagged_Vol'] = data['Volatility_21d'].shift(1)
    data['Lagged_Return'] = data['Returns'].shift(1)
    data['Momentum'] = data['Close'] - data['Close'].shift(5)
    
    # Target (Tomorrow's Volatility)
    data['Target'] = data['Volatility_21d'].shift(-1)
    
    return data.dropna()

# 4. MAIN EXECUTION
if st.sidebar.button("Run Prediction"):
    with st.spinner(f"Fetching data for {ticker}..."):
        df = load_data(ticker, years)
        
        if df is None:
            st.error("Invalid Ticker or No Data Found.")
        else:
            # Prepare data
            model_data = prepare_data(df)
            
            # Split Data
            features = ['Lagged_Vol', 'Lagged_Return', 'Momentum', 'Volume']
            X = model_data[features]
            y = model_data['Target']
            
            split = int(len(model_data) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            # Train Model
# ---------------------------------------------------------
            # NEW: Hyperparameter Tuning (The "Math" Fix)
            # ---------------------------------------------------------
            
            # 1. Define the "Grid" of options to test
            # We are asking the computer: "Try all these combinations and tell me which wins"
            param_grid = {
                'max_depth': [3, 5],              # How deep the tree goes (prevents overfitting)
                'learning_rate': [0.01, 0.05],    # How fast the model learns
                'n_estimators': [100, 200],       # Number of trees
                'subsample': [0.8, 1.0]           # Fraction of data to use per tree
            }

            # 2. Use TimeSeriesSplit
            # CRITICAL FOR FINANCE: Standard Cross-Validation shuffles data. 
            # We cannot do that (can't use 2024 data to predict 2023). 
            # TimeSeriesSplit ensures we only train on PAST to predict FUTURE.
            tscv = TimeSeriesSplit(n_splits=3)

            # 3. Initialize the Search
            # "verbose=0" keeps it quiet, change to 1 to see progress in terminal
            grid_search = GridSearchCV(estimator=xgb.XGBRegressor(), 
                                       param_grid=param_grid, 
                                       cv=tscv, 
                                       scoring='neg_mean_squared_error',
                                       verbose=0)

            # 4. Run the Search (This takes time!)
            if 'st' in globals() or 'streamlit' in str(globals()):
                st.info("⚡️ Tuning Hyperparameters... (This uses Grid Search to find the best math)")
            else:
                print("Tuning Hyperparameters... (This may take a moment)")
            
            grid_search.fit(X_train, y_train)

            # 5. Get the Winner
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Display the winning parameters to the user (Great for interviews!)
            if 'st' in globals() or 'streamlit' in str(globals()):
                st.success(f"Best Model Found! Parameters: {best_params}")
            else:
                print(f"Best Parameters: {best_params}")

            # 6. Predict using the Best Model
            # 6. Predict using the Best Model
            predictions = best_model.predict(X_test)
            
            # ---------------------------------------------------------
            # END OF NEW SECTION
            # ---------------------------------------------------------
            
            # Show Metrics
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            latest_vol = model_data['Volatility_21d'].iloc[-1]
            
            # FIX: Use 'best_model' instead of 'model'
            predicted_vol = best_model.predict(X.iloc[[-1]])[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("Model RMSE (Error)", f"{rmse:.4f}")
            col2.metric("Current Volatility", f"{latest_vol:.4f}")
            col3.metric("Predicted Next Volatility", f"{predicted_vol:.4f}", 
                        delta=f"{predicted_vol - latest_vol:.4f}")

            # Plotting
            st.subheader("Volatility Forecast vs Actual")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_test.index, y_test, label='Actual Volatility', color='blue', alpha=0.5)
            ax.plot(y_test.index, predictions, label='XGBoost Prediction', color='red', alpha=0.8)
            ax.legend()
            st.pyplot(fig)
            
            # Feature Importance
            st.subheader("What is driving the prediction?")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            
            # FIX: Use 'best_model' instead of 'model'
            xgb.plot_importance(best_model, ax=ax2)
            st.pyplot(fig2)