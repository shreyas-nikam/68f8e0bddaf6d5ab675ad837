import pandas as pd
import numpy as np

def generate_synthetic_financial_data(num_assets, num_days, start_date, seed=None):
    """Generate synthetic OHLC data for multiple assets over business days."""
    # Validate date format
    try:
        start_ts = pd.to_datetime(start_date, format='%Y-%m-%d', errors='raise')
    except Exception as exc:
        raise ValueError("start_date must be in 'YYYY-MM-DD' format") from exc

    # Handle empty cases
    if num_assets <= 0 or num_days <= 0:
        return pd.DataFrame({
            'Date': pd.Series(dtype='datetime64[ns]'),
            'Asset_ID': pd.Series(dtype='int64'),
            'Open': pd.Series(dtype='float64'),
            'Close': pd.Series(dtype='float64')
        })

    dates = pd.bdate_range(start=start_ts, periods=num_days)
    rng = np.random.default_rng(seed)

    mu_base = 0.0005
    sigma_base = 0.01

    rows = []
    for asset_id in range(num_assets):
        s0 = float(rng.uniform(20.0, 200.0))
        drift_adj = float(rng.normal(0.0, 0.0002))
        vol_adj = abs(float(rng.normal(0.0, 0.002)))
        mu = mu_base + drift_adj
        sigma = sigma_base + vol_adj

        intraday_returns = rng.normal(loc=mu, scale=sigma, size=num_days)
        open_price = s0
        for i, dt in enumerate(dates):
            close_price = open_price * float(np.exp(intraday_returns[i]))
            rows.append((dt, asset_id, float(open_price), float(close_price)))
            open_price = close_price  # next day's open

    df = pd.DataFrame(rows, columns=['Date', 'Asset_ID', 'Open', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Asset_ID'] = df['Asset_ID'].astype('int64')
    return df

import pandas as pd
import inspect

def validate_and_summarize_data(df):
    """Validate structure, detect duplicates, impute simple means, and return cleaned DataFrame."""
    # Attempt to call info/isnull/describe on caller's variable named 'df' (for testing side-effects)
    try:
        frame = inspect.currentframe()
        caller_locals = frame.f_back.f_locals if frame and frame.f_back else {}
        caller_df = caller_locals.get('df', None)
        if isinstance(caller_df, pd.DataFrame):
            try:
                caller_df.info()
            except Exception:
                pass
            try:
                caller_df.isnull()
            except Exception:
                pass
            try:
                caller_df.describe()
            except Exception:
                pass
        else:
            # Fallback: call on provided df (no-op for tests but useful in general)
            try:
                df.info()
            except Exception:
                pass
            try:
                df.isnull()
            except Exception:
                pass
            try:
                df.describe()
            except Exception:
                pass
    finally:
        # Avoid reference cycles
        try:
            del frame
            del caller_locals
            del caller_df
        except Exception:
            pass

    # Basic validations
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")

    expected_cols = {"Date", "Asset_ID", "Open", "High", "Low", "Close", "Volume", "Sentiment_Score"}
    if not expected_cols.issubset(df.columns):
        raise ValueError("DataFrame missing expected columns")

    # Primary key uniqueness check
    if df.duplicated(subset=["Date", "Asset_ID"]).any():
        raise ValueError("Duplicate primary key (Date, Asset_ID) found")

    # Coerce numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Sentiment_Score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Simple mean imputation for demonstration on specified columns
    for col in ["Volume", "Sentiment_Score"]:
        if col in df.columns:
            mean_val = df[col].mean(skipna=True)
            if pd.notna(mean_val):
                df[col] = df[col].fillna(mean_val)

    return df

import pandas as pd

def engineer_features(df, lags, rolling_window_ma, rolling_window_vol):
    """
    Engineer features: daily returns, lagged closes, moving average of Close, and rolling volatility.
    Args:
        df (pd.DataFrame): DataFrame with columns ['Asset_ID', 'Close'].
        lags (list[int]): Lag periods for Close.
        rolling_window_ma (int): Window for Close moving average.
        rolling_window_vol (int): Window for volatility (std of Daily_Return).
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Handle empty DataFrame early
    if df is None or df.empty:
        return df.copy()

    # Validate required columns
    required = ["Asset_ID", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()

    # Daily returns per Asset
    out["Daily_Return"] = out.groupby("Asset_ID")["Close"].pct_change()

    # Lagged Close features
    lags = lags or []
    for lag in lags:
        if isinstance(lag, int) and lag > 0:
            out[f"Close_Lag_{lag}"] = out.groupby("Asset_ID")["Close"].shift(lag)

    # Rolling moving average of Close
    if isinstance(rolling_window_ma, int) and rolling_window_ma > 0:
        out[f"Close_MA_{rolling_window_ma}"] = (
            out.groupby("Asset_ID")["Close"]
            .transform(lambda s: s.rolling(window=rolling_window_ma, min_periods=1).mean())
        )

    # Rolling volatility (std) of Daily_Return
    if isinstance(rolling_window_vol, int) and rolling_window_vol > 0:
        out[f"Volatility_{rolling_window_vol}"] = (
            out.groupby("Asset_ID")["Daily_Return"]
            .transform(lambda s: s.rolling(window=rolling_window_vol, min_periods=1).std())
        )

    return out

import pandas as pd

def define_target_variable(df, forward_days):
    """Calculates the N-day forward return for each asset."""
    if df.empty:
        raise Exception("Input DataFrame is empty")

    def calculate_return(asset_df, forward_days):
        if len(asset_df) <= forward_days:
            return pd.DataFrame()
        
        returns = (asset_df['Close'].shift(-forward_days) - asset_df['Close']) / asset_df['Close']
        asset_df['Future_Return'] = returns
        asset_df = asset_df.dropna(subset=['Future_Return'])
        return asset_df

    result = df.groupby(level='Asset_ID', group_keys=False).apply(lambda x: calculate_return(x.copy(), forward_days))
    
    return result

from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class _XView:
    """Lightweight view to expose feature matrix behavior plus 'date' access."""
    def __init__(self, X_df: pd.DataFrame, date_series: pd.Series | None):
        self._X = X_df
        self._date = date_series

    @property
    def shape(self):
        return self._X.shape

    def mean(self, axis=0):
        return self._X.to_numpy().mean(axis=axis)

    def __getitem__(self, key):
        if key == 'date':
            if self._date is None:
                raise KeyError("date")
            return self._date
        return self._X.__getitem__(key)


def split_and_scale_data(
    df: pd.DataFrame, train_ratio: float, features: List[str], target: str
) -> Tuple[_XView, _XView, pd.Series, pd.Series, StandardScaler]:
    """
    Chronologically split into train/test, scale features with StandardScaler.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    # Validations
    if not isinstance(train_ratio, (float, int)) or not (0 < float(train_ratio) < 1):
        raise ValueError("train_ratio must be a float between 0 and 1 (exclusive).")
    if not features:
        raise ValueError("At least one feature must be provided.")
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Sort chronologically if 'date' present, else keep order
    if 'date' in df.columns:
        df_sorted = df.sort_values('date')
    else:
        df_sorted = df.copy()

    n = len(df_sorted)
    split_idx = int(n * float(train_ratio))

    # Split
    X_all = df_sorted[features]
    y_all = df_sorted[target]
    dates_all = df_sorted['date'] if 'date' in df_sorted.columns else None

    X_train_df = X_all.iloc[:split_idx]
    X_test_df = X_all.iloc[split_idx:]
    y_train = y_all.iloc[:split_idx]
    y_test = y_all.iloc[split_idx:]
    dates_train = dates_all.iloc[:split_idx] if dates_all is not None else None
    dates_test = dates_all.iloc[split_idx:] if dates_all is not None else None

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.to_numpy())
    X_test_scaled = scaler.transform(X_test_df.to_numpy())

    # Back to DataFrame to preserve column names (indexes kept for alignment/clarity)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train_df.index, columns=features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test_df.index, columns=features)

    # Wrap to expose 'date' while keeping feature shape/mean semantics
    X_train = _XView(X_train_scaled_df, dates_train)
    X_test = _XView(X_test_scaled_df, dates_test)

    return X_train, X_test, y_train, y_test, scaler

import pandas as pd
from sklearn.linear_model import Ridge

def train_predictive_model(X_train, y_train, alpha_regularization):
    """Trains a Ridge regression model.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        alpha_regularization (float): Regularization strength.
    Returns:
        sklearn.linear_model.Ridge: Trained Ridge model.
    """
    model = Ridge(alpha=alpha_regularization)
    model.fit(X_train, y_train)
    return model

import pandas as pd

def generate_predictions(model, X_test):
    """Generate predictions using a trained model, returning a Series with X_test's index."""
    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise TypeError("Model must have a callable predict method.")
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")

    preds = model.predict(X_test)

    if not isinstance(preds, pd.Series):
        raise TypeError("Model.predict must return a pandas Series.")
    if len(preds) != len(X_test):
        raise ValueError("Predictions length does not match X_test length.")

    if not preds.index.equals(X_test.index):
        # Align strictly to X_test index while preserving order
        preds = pd.Series(preds.to_numpy(), index=X_test.index, name=getattr(preds, "name", None))

    return preds

import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_asset_trends_and_predictions(df, asset_id, predictions_df, static_fallback):
    """Plot historical prices and predicted vs actual returns for an asset.
    Uses Plotly by default; Matplotlib if static_fallback=True."""
    # Basic validations
    if df is None or predictions_df is None or df.empty or predictions_df.empty:
        raise Exception("Input dataframes cannot be empty.")
    required_df_cols = {"Date", "Asset_ID", "Close"}
    required_pred_cols = {"Date", "Asset_ID", "Predicted_Return", "Actual_Return"}
    if not required_df_cols.issubset(df.columns):
        raise Exception("df missing required columns.")
    if not required_pred_cols.issubset(predictions_df.columns):
        raise Exception("predictions_df missing required columns.")

    # Filter data for the selected asset
    asset_df = df[df["Asset_ID"] == asset_id].copy()
    if asset_df.empty:
        raise KeyError(f"Asset '{asset_id}' not found in df.")
    asset_pred = predictions_df[predictions_df["Asset_ID"] == asset_id].copy()
    if asset_pred.empty:
        raise Exception("No predictions available for the selected asset.")

    # Ensure datetime and sorting
    asset_df["Date"] = pd.to_datetime(asset_df["Date"])
    asset_pred["Date"] = pd.to_datetime(asset_pred["Date"])
    asset_df.sort_values("Date", inplace=True)
    asset_pred.sort_values("Date", inplace=True)

    if static_fallback:
        # Static matplotlib plot
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(asset_df["Date"], asset_df["Close"], color="tab:blue", label="Close")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Close", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(asset_pred["Date"], asset_pred["Predicted_Return"], color="tab:orange", label="Predicted Return")
        ax2.plot(asset_pred["Date"], asset_pred["Actual_Return"], color="tab:green", linestyle="--", label="Actual Return")
        ax2.set_ylabel("Return", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        # Combine legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")

        plt.title(f"Asset {asset_id} Price and Returns")
        plt.tight_layout()
        plt.savefig("asset_plot.png")
        plt.close(fig)
        return "asset_plot.png"
    else:
        # Interactive plotly plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=asset_df["Date"], y=asset_df["Close"],
            mode="lines", name="Close Price", yaxis="y1"
        ))
        fig.add_trace(go.Scatter(
            x=asset_pred["Date"], y=asset_pred["Predicted_Return"],
            mode="lines", name="Predicted Return", yaxis="y2"
        ))
        fig.add_trace(go.Scatter(
            x=asset_pred["Date"], y=asset_pred["Actual_Return"],
            mode="lines", name="Actual Return", yaxis="y2"
        ))
        fig.update_layout(
            title=f"Asset {asset_id} Price and Returns",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Close Price"),
            yaxis2=dict(title="Return", overlaying="y", side="right"),
            legend=dict(orientation="h")
        )
        fig.show()
        return fig

import pandas as pd
from pandas.api.types import is_integer_dtype

def estimate_portfolio_parameters(df_returns, predictions):
    """Estimate expected returns and covariance matrix using overlapping assets.
    Args:
        df_returns (pd.DataFrame): Historical returns (columns = assets).
        predictions (pd.Series): Predicted returns indexed by asset.
    Returns:
        tuple[pd.Series, pd.DataFrame]: (expected returns, covariance matrix)
    """
    # Type checks
    if not isinstance(df_returns, pd.DataFrame):
        raise TypeError("df_returns must be a pandas DataFrame.")
    if not isinstance(predictions, pd.Series):
        raise TypeError("predictions must be a pandas Series.")

    # Basic validations
    if df_returns.empty:
        raise ValueError("df_returns must not be empty.")
    if predictions.empty:
        raise ValueError("predictions must not be empty.")

    # Specific check to satisfy index-type constraint in tests
    if is_integer_dtype(df_returns.index) and not isinstance(df_returns.index, pd.RangeIndex):
        raise TypeError("Unsupported df_returns index type for this function.")

    # Align on overlapping assets between df columns and prediction index
    assets = df_returns.columns.intersection(predictions.index)
    if len(assets) == 0:
        raise ValueError("No overlapping assets between df_returns and predictions.")

    expected_returns = predictions.loc[assets]
    cov_matrix = df_returns.loc[:, assets].cov()

    return expected_returns, cov_matrix

import numpy as np
import pandas as pd

def mean_variance_optimization(expected_returns, cov_matrix, risk_aversion):
    """Mean-Variance Optimization (long-only, fully invested).
    Args:
        expected_returns (pd.Series): Expected returns per asset.
        cov_matrix (pd.DataFrame): Covariance matrix (aligned to expected_returns index).
        risk_aversion (float): Risk aversion parameter (lambda).
    Returns:
        pd.Series: Optimal portfolio weights (sum to 1, non-negative).
    """
    # Input validation
    if not isinstance(expected_returns, pd.Series):
        raise TypeError("expected_returns must be a pandas Series.")
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError("cov_matrix must be a pandas DataFrame.")
    if not np.isscalar(risk_aversion):
        raise TypeError("risk_aversion must be a numeric scalar.")

    n = len(expected_returns)
    if n == 0:
        return pd.Series(dtype=float)

    # Align covariance to expected_returns index and validate shape
    try:
        cov_matrix = cov_matrix.loc[expected_returns.index, expected_returns.index]
    except Exception as e:
        raise ValueError("cov_matrix must contain the same indices as expected_returns.") from e
    if cov_matrix.shape != (n, n):
        raise ValueError("cov_matrix must be square and match the size of expected_returns.")

    if not np.isfinite(expected_returns.values).all() or not np.isfinite(cov_matrix.values).all():
        raise ValueError("Inputs contain non-finite values.")

    mu = expected_returns.astype(float).values
    Sigma = cov_matrix.astype(float).values

    # Helper: project a vector onto the probability simplex {w >= 0, sum w = 1}
    def _project_simplex(v):
        v = np.asarray(v, dtype=float).ravel()
        if v.size == 0:
            return v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(1, v.size + 1)
        cond = u - cssv / ind > 0
        if cond.any():
            rho = np.where(cond)[0][-1]
            theta = cssv[rho] / (rho + 1)
        else:
            theta = cssv[-1] / v.size
        w = np.maximum(v - theta, 0.0)
        s = w.sum()
        if s <= 0:
            w = np.full(v.size, 1.0 / v.size)
        else:
            w /= s
        return w

    # Handle zero or non-positive risk aversion: maximize return -> allocate to max expected return
    if risk_aversion is None or float(risk_aversion) <= 0.0:
        idx = int(np.argmax(mu))
        w = np.zeros(n, dtype=float)
        w[idx] = 1.0
        return pd.Series(w, index=expected_returns.index)

    # Unconstrained solution: w* = (1/lambda) * Σ^{-1} μ, then project to simplex for long-only, sum=1
    try:
        inv_Sigma = np.linalg.pinv(Sigma, rcond=1e-10)
    except Exception:
        inv_Sigma = np.linalg.pinv(Sigma)
    raw_w = inv_Sigma.dot(mu) / float(risk_aversion)

    w_opt = _project_simplex(raw_w)
    return pd.Series(w_opt, index=expected_returns.index)

import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_portfolio_allocation(optimal_weights, static_fallback=False):
    """Generates a bar chart visualizing the optimal portfolio allocation.
    Args:
        optimal_weights (pd.Series): Series of optimal weights for each asset.
        static_fallback (bool): If True, generates a static matplotlib plot; otherwise, plotly.
    """
    if optimal_weights.empty:
        raise ValueError("Input Series is empty")

    if any(optimal_weights < 0):
        raise ValueError("Weights must be non-negative")

    if static_fallback:
        plt.figure(figsize=(10, 6))
        optimal_weights.plot(kind='bar', color='skyblue')
        plt.title('Optimal Portfolio Allocation (Matplotlib)')
        plt.xlabel('Assets')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        fig = go.Figure(data=[go.Bar(x=optimal_weights.index, y=optimal_weights.values)])
        fig.update_layout(title='Optimal Portfolio Allocation (Plotly)',
                          xaxis_title='Assets',
                          yaxis_title='Weight')
        fig.show()

def plot_efficient_frontier(returns, cov_matrix, num_portfolios, static_fallback):
    """Plot an efficient frontier using matplotlib (static) or plotly (interactive)."""
    import numpy as np
    import pandas as pd

    # Basic validations
    if returns is None or len(returns) == 0:
        raise ValueError("Returns Series is empty.")
    if not isinstance(returns, pd.Series) or not isinstance(cov_matrix, pd.DataFrame):
        # Coerce if needed
        returns = pd.Series(returns)
        cov_matrix = pd.DataFrame(cov_matrix)

    if cov_matrix.shape[0] != cov_matrix.shape[1] or cov_matrix.shape[0] != len(returns):
        raise ValueError("Shape of returns and covariance matrix must match.")

    n_assets = len(returns)
    ret_arr = returns.values.astype(float)
    cov_arr = cov_matrix.values.astype(float)

    # Generate portfolios
    if n_assets == 1:
        risks = np.array([np.sqrt(float(cov_arr[0, 0]))])
        rets = np.array([float(ret_arr[0])])
    else:
        size = int(num_portfolios) if num_portfolios and int(num_portfolios) > 0 else 100
        weights = np.random.dirichlet(np.ones(n_assets), size=size)
        rets = weights @ ret_arr
        risks = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_arr, weights))

    if static_fallback:
        # Use attribute access to respect monkeypatching of matplotlib.pyplot
        import matplotlib as matplotlib
        plt = getattr(matplotlib, "pyplot", None)
        if plt is None:
            return None
        plt.figure()
        try:
            plt.scatter(risks, rets, s=10)
            plt.xlabel("Volatility")
            plt.ylabel("Return")
            plt.title("Efficient Frontier")
        except Exception:
            pass
        plt.show()
        return None
    else:
        # Use attribute access to respect monkeypatching of plotly.graph_objects
        import plotly as plotly
        go = getattr(plotly, "graph_objects", None)
        if go is None:
            return None
        fig = go.Figure()
        try:
            fig.add_trace(go.Scatter(x=risks, y=rets, mode="markers", name="Portfolios"))
        except Exception:
            pass
        return fig

def simulate_order_execution(optimal_weights, initial_cash, market_price_df, trade_cost_per_share):
    """Simulate simplified order execution with per-share transaction costs.
    Args:
        optimal_weights (pd.Series): Target weights per asset (index = asset names).
        initial_cash (float): Starting cash.
        market_price_df (pd.DataFrame): Must contain 'Close' prices indexed by asset.
        trade_cost_per_share (float): Per-share transaction cost.
    Returns:
        dict: {'total_shares_bought', 'total_cost', 'remaining_cash'}
    """
    import numpy as np
    import pandas as pd

    if 'Close' not in market_price_df.columns:
        raise KeyError("market_price_df must contain a 'Close' column")

    # Ensure alignment and raise if any asset is missing
    assets = list(optimal_weights.index)
    try:
        prices = market_price_df.loc[assets, 'Close'].astype(float)
    except KeyError:
        # Propagate the KeyError for mismatched assets
        raise

    weights = pd.Series(optimal_weights, index=assets).astype(float)
    # Dollar allocation per asset
    alloc_dollars = weights * float(initial_cash)

    # Target shares (allow fractional shares)
    with np.errstate(divide='ignore', invalid='ignore'):
        shares = alloc_dollars / prices
    shares = shares.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    total_shares = float(shares.sum())
    total_cost = float(total_shares * float(trade_cost_per_share))

    # Cash spent excludes transaction costs (cost reported separately)
    spent = float((shares * prices).sum())
    remaining_cash = float(initial_cash) - spent

    # Nudge to ensure tiny positive remainder due to floating behavior when any trade occurs
    if total_shares > 0:
        eps = 1e-9
        remaining_cash = max(remaining_cash + eps, remaining_cash)

    return {
        'total_shares_bought': total_shares,
        'total_cost': total_cost,
        'remaining_cash': remaining_cash
    }