import pytest
from definition_ffe522b3ab2845e8a8e95e5ecbb7d188 import simulate_order_execution
import pandas as pd

def test_simulate_order_execution_no_shares():
    optimal_weights = pd.Series([0, 0, 0], index=['Asset1', 'Asset2', 'Asset3'])
    initial_cash = 100000
    market_price_df = pd.DataFrame({'Close': [10, 20, 30]}, index=['Asset1', 'Asset2', 'Asset3'])
    trade_cost_per_share = 0.01
    result = simulate_order_execution(optimal_weights, initial_cash, market_price_df, trade_cost_per_share)
    assert result['total_shares_bought'] == 0
    assert result['total_cost'] == 0
    assert result['remaining_cash'] == 100000

def test_simulate_order_execution_single_asset():
    optimal_weights = pd.Series([1], index=['Asset1'])
    initial_cash = 100000
    market_price_df = pd.DataFrame({'Close': [10]}, index=['Asset1'])
    trade_cost_per_share = 0.01
    result = simulate_order_execution(optimal_weights, initial_cash, market_price_df, trade_cost_per_share)
    shares = (initial_cash / 10)
    cost = shares*0.01
    assert abs(result['total_shares_bought'] - (initial_cash / 10)) < 1e-6
    assert abs(result['total_cost'] - cost) < 1e-6
    assert abs(result['remaining_cash'] - 0) < 1

def test_simulate_order_execution_multiple_assets():
    optimal_weights = pd.Series([0.2, 0.3, 0.5], index=['Asset1', 'Asset2', 'Asset3'])
    initial_cash = 100000
    market_price_df = pd.DataFrame({'Close': [10, 20, 30]}, index=['Asset1', 'Asset2', 'Asset3'])
    trade_cost_per_share = 0.01
    result = simulate_order_execution(optimal_weights, initial_cash, market_price_df, trade_cost_per_share)
    expected_shares_asset1 = (0.2 * initial_cash) / 10
    expected_shares_asset2 = (0.3 * initial_cash) / 20
    expected_shares_asset3 = (0.5 * initial_cash) / 30

    expected_cost_asset1 = expected_shares_asset1*trade_cost_per_share
    expected_cost_asset2 = expected_shares_asset2*trade_cost_per_share
    expected_cost_asset3 = expected_shares_asset3*trade_cost_per_share
    expected_total_cost = expected_cost_asset1 + expected_cost_asset2 + expected_cost_asset3
    assert abs(result['total_shares_bought'] - (expected_shares_asset1+expected_shares_asset2+expected_shares_asset3)) < 1
    assert abs(result['total_cost'] - expected_total_cost) < 1e-6
    assert result['remaining_cash'] > 0

def test_simulate_order_execution_high_trade_cost():
    optimal_weights = pd.Series([1], index=['Asset1'])
    initial_cash = 10000
    market_price_df = pd.DataFrame({'Close': [10]}, index=['Asset1'])
    trade_cost_per_share = 1  # High trade cost, almost 10% of the initial portfolio
    result = simulate_order_execution(optimal_weights, initial_cash, market_price_df, trade_cost_per_share)
    shares = (initial_cash / 10)
    cost = shares
    assert abs(result['total_shares_bought'] - (initial_cash / 10)) < 1e-6
    assert abs(result['total_cost'] - cost ) < 1e-6
    assert result['remaining_cash'] > 0

def test_simulate_order_execution_mismatched_assets():
    optimal_weights = pd.Series([0.5, 0.5], index=['Asset1', 'Asset2'])
    initial_cash = 100000
    market_price_df = pd.DataFrame({'Close': [10]}, index=['Asset1'])  # Missing Asset2
    trade_cost_per_share = 0.01
    with pytest.raises(KeyError):
        simulate_order_execution(optimal_weights, initial_cash, market_price_df, trade_cost_per_share)