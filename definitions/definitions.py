import pandas as pd
import numpy as np

def generate_synthetic_financial_data(num_assets, num_days, start_date, seed=None):
    """Generates synthetic financial time-series data."""
    try:
        start_date = pd.to_datetime(start_date)
    except ValueError:
        raise ValueError("Invalid date format. Please use 'YYYY-MM-DD'.")

    if seed is not None:
        np.random.seed(seed)

    data = []
    for asset_id in range(num_assets):
        for i in range(num_days):
            date = start_date + pd.Timedelta(days=i)
            open_price = np.random.rand() * 100
            high_price = open_price + np.random.rand() * 20
            low_price = open_price - np.random.rand() * 20
            close_price = open_price + np.random.randn() * 10
            volume = np.random.randint(1000, 10000)
            sentiment_score = np.random.uniform(-1, 1)

            data.append([date, asset_id, open_price, high_price, low_price, close_price, volume, sentiment_score])

    df = pd.DataFrame(data, columns=['Date', 'Asset_ID', 'Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment_Score'])
    return df

import pandas as pd
import numpy as np

def validate_and_summarize_data(df):
    """Validates data, imputes missing values, and checks for duplicates."""

    if df.empty:
        raise Exception("DataFrame is empty.")

    required_columns = ['Date', 'Asset_ID', 'Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment_Score']
    if not all(col in df.columns for col in required_columns):
        raise Exception("DataFrame does not contain the required columns")

    primary_key_cols = ['Date', 'Asset_ID']
    if not df.duplicated(subset=primary_key_cols).any():
        pass
    else:
        raise Exception("DataFrame does not have unique primary key")

    cols_to_impute = ['Volume', 'Sentiment_Score']
    for col in cols_to_impute:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    return df

import pandas as pd
import numpy as np

def engineer_features(df, lags, rolling_window_ma, rolling_window_vol):
    """Engineers features like daily returns, lagged returns, moving averages, and rolling volatility.
    Args:
        df (pd.DataFrame): The input DataFrame with 'Close' prices and 'Asset_ID'.
        lags (list): List of lag periods for return features.
        rolling_window_ma (int): Window for rolling moving average.
        rolling_window_vol (int): Window for rolling volatility.
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    if df.empty:
        return df

    df['daily_return'] = df.groupby('Asset_ID')['Close'].pct_change()

    for lag in lags:
        df[f'lagged_return_{lag}'] = df.groupby('Asset_ID')['daily_return'].shift(lag)

    df['moving_average_20'] = df.groupby('Asset_ID')['Close'].rolling(window=max(1,rolling_window_ma)).mean().values
    df['rolling_volatility_20'] = df.groupby('Asset_ID')['daily_return'].rolling(window=max(1,rolling_window_vol)).std().values
    df.fillna(0, inplace=True)
    return df

import pandas as pd


def define_target_variable(df, forward_days):
    """Calculates the N-day forward return for each asset."""
    
    def calculate_return(group):
        if len(group) > forward_days:
            group['Future_Return'] = (group['Close'].shift(-forward_days) - group['Close']) / group['Close']
            group = group.dropna(subset=['Future_Return'])
            return group
        else:
            return pd.DataFrame()

    df = df.groupby('Asset_ID').apply(calculate_return)
    df = df.reset_index(drop=True)
    
    if not df.empty:
        df = df[df['Close'] != 0]
    
    if not df.empty:
        df = df.dropna(subset=['Future_Return'])

    return df

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_and_scale_data(df, train_ratio, features, target):
    """Splits and scales data into train/test sets."""
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if not 0 < train_ratio < 1:
        raise ValueError("Train ratio must be between 0 and 1.")
    if not features:
        raise ValueError("Features list cannot be empty.")
    if target not in df.columns:
        raise KeyError("Target column not found in DataFrame.")

    train_size = int(len(df) * train_ratio)
    train_df = df[:train_size]
    test_df = df[train_size:]

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df[target].copy()
    y_test = test_df[target].copy()

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test, scaler

import pandas as pd
from sklearn.linear_model import Ridge


def train_predictive_model(X_train, y_train, alpha_regularization):
    """Trains a Ridge regression model.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        alpha_regularization (float): Regularization strength (alpha).
    Returns:
        sklearn.linear_model.Ridge: Trained Ridge model.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("Training data cannot be empty.")

    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train must have the same index.")

    model = Ridge(alpha=alpha_regularization)
    model.fit(X_train, y_train)
    return model

import pandas as pd

def generate_predictions(model, X_test):
    """
    Generates predictions using the trained model.

    Arguments:
        model (sklearn.base.BaseEstimator): Trained predictive model.
        X_test (pd.DataFrame): Testing features.

    Output:
        pd.Series: Predicted target values.
    """
    try:
        predictions = pd.Series(model.predict(X_test))
    except Exception:
        predictions = pd.Series([float('NaN')] * len(X_test))

    return predictions

import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_asset_trends_and_predictions(df, asset_id, predictions_df, static_fallback):
    """Generates an interactive line plot of historical prices and predicted future returns for a selected asset. Provides a static matplotlib fallback.
    Arguments: df (pd.DataFrame): Original DataFrame with historical prices. asset_id (str): The ID of the asset to plot. predictions_df (pd.DataFrame): DataFrame containing actual and predicted returns. static_fallback (bool): If True, generates a static matplotlib plot; otherwise, plotly.
    """
    asset_df = df[df['Asset_ID'] == asset_id].copy()

    if asset_df.empty:
        return

    if static_fallback:
        plt.figure(figsize=(12, 6))
        plt.plot(asset_df['Date'], asset_df['Close'], label='Historical Prices')

        predictions_asset_df = predictions_df[predictions_df['Asset_ID'] == asset_id].copy()

        if not predictions_asset_df.empty:
            plt.plot(predictions_asset_df['Date'], asset_df[asset_df['Date'].isin(predictions_asset_df['Date'])]['Close'], marker='o', linestyle='None', label='Actual Return Prices')
            plt.plot(predictions_asset_df['Date'], asset_df[asset_df['Date'].isin(predictions_asset_df['Date'])]['Close'], marker='x', linestyle='None', label='Predicted Return Prices')

        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Asset Trends and Predictions for {asset_id}')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=asset_df['Date'], y=asset_df['Close'], mode='lines', name='Historical Prices'))

        predictions_asset_df = predictions_df[predictions_df['Asset_ID'] == asset_id].copy()

        if not predictions_asset_df.empty:
            actual_dates = predictions_asset_df['Date'].tolist()
            actual_prices = asset_df[asset_df['Date'].isin(actual_dates)]['Close'].tolist()
            fig.add_trace(go.Scatter(x=actual_dates, y=actual_prices, mode='markers', name='Actual Returns'))
            
            
            predicted_dates = predictions_asset_df['Date'].tolist()
            predicted_prices = asset_df[asset_df['Date'].isin(predicted_dates)]['Close'].tolist()

        fig.update_layout(title=f'Asset Trends and Predictions for {asset_id}', xaxis_title='Date', yaxis_title='Price')
        fig.show()

import pandas as pd
import numpy as np
from scipy.optimize import minimize

def mean_variance_optimization(expected_returns, cov_matrix, risk_aversion):
    """Performs Mean-Variance Optimization."""
    num_assets = len(expected_returns)
    initial_weights = np.array([1/num_assets] * num_assets)
    bounds = [(0, 1)] * num_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    def objective_function(weights, expected_returns, cov_matrix, risk_aversion):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return + risk_aversion * portfolio_volatility**2

    result = minimize(objective_function, initial_weights, args=(expected_returns, cov_matrix, risk_aversion),
                       method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = pd.Series(result.x, index=expected_returns.index)
    return optimal_weights

import pandas as pd


def estimate_portfolio_parameters(df_returns, predictions):
    """Estimates expected returns and covariance matrix.

    Args:
        df_returns (pd.DataFrame): Historical returns.
        predictions (pd.Series): Predicted returns.

    Returns:
        tuple: Expected returns (pd.Series), covariance matrix (pd.DataFrame).
    """
    if df_returns.empty or predictions.empty:
        raise ValueError("DataFrame and Series should not be empty.")

    if predictions is None:
        raise TypeError("Predictions cannot be None.")

    if not set(df_returns.columns) == set(predictions.index):
        raise ValueError("Assets in DataFrame and Series do not match.")

    expected_returns = predictions
    cov_matrix = df_returns.cov()

    return expected_returns, cov_matrix

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as py

def plot_portfolio_allocation(optimal_weights, static_fallback):
    """Generates a bar chart visualizing the optimal portfolio allocation.
    Args:
        optimal_weights (pd.Series): Series of optimal weights for each asset.
        static_fallback (bool): If True, generates a static matplotlib plot; otherwise, plotly.
    """
    if not isinstance(optimal_weights, pd.Series):
        raise TypeError("optimal_weights must be a pandas Series.")

    if optimal_weights.empty:
        print("Optimal weights are empty, no plot to generate.")
        return

    if static_fallback:
        plt.figure(figsize=(10, 6))
        optimal_weights.plot(kind='bar')
        plt.title('Optimal Portfolio Allocation')
        plt.xlabel('Assets')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        data = [go.Bar(x=optimal_weights.index, y=optimal_weights.values)]
        layout = go.Layout(title='Optimal Portfolio Allocation',
                           xaxis=dict(title='Assets'),
                           yaxis=dict(title='Weight'))
        fig = go.Figure(data=data, layout=layout)
        py.iplot(fig)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_efficient_frontier(returns, cov_matrix, num_portfolios, static_fallback):
    """
    Calculates and plots the efficient frontier.
    Args:
        returns (pd.Series): Expected returns for each asset.
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
        num_portfolios (int): Number of random portfolios to generate.
        static_fallback (bool): Use matplotlib if True, otherwise plotly.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError("cov_matrix must be a pandas DataFrame")
    if returns.empty or cov_matrix.empty:
        raise Exception("Returns or Covariance matrix cannot be empty")

    num_assets = len(returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.sum(returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = (portfolio_return - 0.0) / portfolio_std_dev

    if static_fallback:
        plt.figure(figsize=(12, 8))
        plt.scatter(results[1, :], results[0, :], c=results[2, :], marker='o', cmap='viridis')
        plt.title('Efficient Frontier (Static)')
        plt.xlabel('Volatility (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.show()
    else:
        fig = go.Figure(data=[go.Scatter(x=results[1, :], y=results[0, :], mode='markers',
                                         marker=dict(color=results[2, :], showscale=True,
                                                     colorbar=dict(title='Sharpe Ratio'),
                                                     colorscale='Viridis'))])

        fig.update_layout(title='Efficient Frontier',
                          xaxis_title='Volatility (Standard Deviation)',
                          yaxis_title='Expected Return')
        fig.show()

import pandas as pd

def simulate_order_execution(optimal_weights, initial_cash, market_price_df, trade_cost_per_share):
    """Simulates order execution with transaction costs."""

    if initial_cash < 0:
        raise ValueError("Initial cash cannot be negative.")

    total_shares_bought = 0
    total_cost = 0
    remaining_cash = initial_cash

    for asset, weight in optimal_weights.items():
        if weight > 0:
            price = market_price_df.loc[asset, 'Close']
            target_amount = initial_cash * weight
            num_shares = target_amount // price
            cost_for_shares = num_shares * price
            trade_cost = num_shares * trade_cost_per_share
            total_asset_cost = cost_for_shares + trade_cost

            if total_asset_cost <= remaining_cash:
                total_shares_bought += num_shares
                total_cost += total_asset_cost
                remaining_cash -= total_asset_cost
            else:
                num_shares = (remaining_cash // (price + trade_cost_per_share)) # Buy maximum possible
                cost_for_shares = num_shares * price
                trade_cost = num_shares * trade_cost_per_share
                total_asset_cost = cost_for_shares + trade_cost

                total_shares_bought += num_shares
                total_cost += total_asset_cost
                remaining_cash -= total_asset_cost
                
    return {
        'total_shares_bought': total_shares_bought,
        'total_cost': total_cost,
        'remaining_cash': remaining_cash
    }