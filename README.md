# market-volatility-predictor
Real-time financial risk forecasting engine using Python, XGBoost, and Streamlit. Features hyperparameter tuning and volatility clustering analysis.
# Market Volatility Prediction Engine

A machine learning pipeline that forecasts stock market volatility using **XGBoost** and **Technical Analysis**. This tool identifies "Volatility Clustering" and predicts future risk regimes to aid in portfolio management.

### Key Features
* **Real-Time Data Pipeline:** Fetches live OHLCV data via Yahoo Finance API.
* **Advanced Feature Engineering:** Calculates Rolling Volatility, RSI, Momentum, and Lagged Returns.
* **XGBoost Regressor:** Optimized using **GridSearchCV** to find the best hyperparameters.
* **Financial Validation:** Uses **TimeSeriesSplit** (not random K-Fold) to prevent data leakage.
* **Interactive Dashboard:** Built with Streamlit for real-time visualization.

### Tech Stack
* **Python 3.9+**
* **Machine Learning:** XGBoost, Scikit-Learn (GridSearch, TimeSeriesSplit)
* **Data Engineering:** Pandas, NumPy, YFinance
* **Visualization:** Matplotlib, Streamlit


### ⚙️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone (https://github.com/Ironyash46/market-volatility-predictor.git)
   
