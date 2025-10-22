import pytest
import pandas as pd
from definition_fdb3ab4fcdb34dc7b9abd5ccda1374bf import plot_portfolio_allocation

def test_plot_portfolio_allocation_empty_weights():
    """Test with empty optimal weights."""
    optimal_weights = pd.Series({})
    try:
        plot_portfolio_allocation(optimal_weights, static_fallback=True)
    except Exception as e:
        assert False, f"Exception raised with empty weights: {e}"

def test_plot_portfolio_allocation_single_asset():
    """Test with only one asset in the portfolio."""
    optimal_weights = pd.Series({'Asset1': 1.0})
    try:
        plot_portfolio_allocation(optimal_weights, static_fallback=True)
    except Exception as e:
        assert False, f"Exception raised with single asset: {e}"

def test_plot_portfolio_allocation_multiple_assets():
    """Test with multiple assets and positive weights."""
    optimal_weights = pd.Series({'Asset1': 0.3, 'Asset2': 0.7})
    try:
        plot_portfolio_allocation(optimal_weights, static_fallback=True)
    except Exception as e:
        assert False, f"Exception raised with multiple assets: {e}"

def test_plot_portfolio_allocation_negative_weights():
    """Test when there is a negative weight."""
    optimal_weights = pd.Series({'Asset1': -0.2, 'Asset2': 1.2})
    try:
        plot_portfolio_allocation(optimal_weights, static_fallback=True)
    except Exception as e:
        assert False, f"Exception raised with single asset: {e}"

def test_plot_portfolio_allocation_non_series_input():
    """Test when the input is not a series."""
    with pytest.raises(TypeError):
        plot_portfolio_allocation([1, 2, 3], static_fallback=True)
