import pytest
from definition_3a1f75de40884b218d8064e8d2186c62 import simulate_order_execution
import pandas as pd

def test_simulate_order_execution_no_trades():
    """Test when optimal weights are all zero, resulting in no trades."""
    optimal_weights = pd.Series({'Asset1': 0.0, 'Asset2': 0.0})
    initial_cash = 100000.0
    market_price_df = pd.DataFrame({'Close': [10.0, 20.0]}, index=['Asset1', 'Asset2'])
    trade_cost_per_share = 0.01
    result = simulate_order_execution(optimal_weights, initial_cash, market_price_df, trade_cost_per_share)
    assert result['total_shares_bought'] == 0
    assert result['total_cost'] == 0
    assert result['remaining_cash'] == initial_cash

def test_simulate_order_execution_single_asset():
    """Test when only one asset is traded."""
    optimal_weights = pd.Series({'Asset1': 1.0, 'Asset2': 0.0})
    initial_cash = 100000.0
    market_price_df = pd.DataFrame({'Close': [10.0, 20.0]}, index=['Asset1', 'Asset2'])
    trade_cost_per_share = 0.01
    result = simulate_order_execution(optimal_weights, initial_cash, market_price_df, trade_cost_per_share)
    assert result['total_shares_bought'] > 0
    assert result['total_cost'] > 0
    assert result['remaining_cash'] < initial_cash

def test_simulate_order_execution_multiple_assets():
    """Test when multiple assets are traded."""
    optimal_weights = pd.Series({'Asset1': 0.5, 'Asset2': 0.5})
    initial_cash = 100000.0
    market_price_df = pd.DataFrame({'Close': [10.0, 20.0]}, index=['Asset1', 'Asset2'])
    trade_cost_per_share = 0.01
    result = simulate_order_execution(optimal_weights, initial_cash, market_price_df, trade_cost_per_share)
    assert result['total_shares_bought'] > 0
    assert result['total_cost'] > 0
    assert result['remaining_cash'] < initial_cash

def test_simulate_order_execution_high_trade_cost():
    """Test with high trade cost to see its impact."""
    optimal_weights = pd.Series({'Asset1': 0.5, 'Asset2': 0.5})
    initial_cash = 100000.0
    market_price_df = pd.DataFrame({'Close': [10.0, 20.0]}, index=['Asset1', 'Asset2'])
    trade_cost_per_share = 1.0
    result = simulate_order_execution(optimal_weights, initial_cash, market_price_df, trade_cost_per_share)
    assert result['total_cost'] > 0

def test_simulate_order_execution_invalid_input():
    """Test with invalid input (e.g., negative cash)."""
    optimal_weights = pd.Series({'Asset1': 0.5, 'Asset2': 0.5})
    initial_cash = -100000.0
    market_price_df = pd.DataFrame({'Close': [10.0, 20.0]}, index=['Asset1', 'Asset2'])
    trade_cost_per_share = 0.01
    with pytest.raises(ValueError):
        simulate_order_execution(optimal_weights, initial_cash, market_price_df, trade_cost_per_share)