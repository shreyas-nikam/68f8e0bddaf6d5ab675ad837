import pytest
import pandas as pd
from unittest.mock import patch
from definition_eb15b8dc48f64b35b0af624666fcfd10 import plot_portfolio_allocation
import matplotlib.pyplot as plt
import plotly.graph_objects as go

@pytest.fixture
def mock_plotly_show(monkeypatch):
    """Mocks plotly.graph_objects.Figure.show to prevent actual plot display."""
    def no_op():
        pass
    monkeypatch.setattr(go.Figure, "show", no_op)

def test_empty_weights(mock_plotly_show):
    """Test with an empty Series of weights."""
    optimal_weights = pd.Series([], dtype='float64')
    with pytest.raises(ValueError) as excinfo:
        plot_portfolio_allocation(optimal_weights)
    assert "Input Series is empty" in str(excinfo.value)


def test_single_asset(mock_plotly_show):
    """Test with only one asset in the portfolio."""
    optimal_weights = pd.Series({'Asset1': 1.0})
    plot_portfolio_allocation(optimal_weights)


def test_negative_weights(mock_plotly_show):
    """Test with negative weights (should not happen in a long-only portfolio)."""
    optimal_weights = pd.Series({'Asset1': 0.3, 'Asset2': -0.2, 'Asset3': 0.9})
    with pytest.raises(ValueError) as excinfo:
        plot_portfolio_allocation(optimal_weights)
    assert "Weights must be non-negative" in str(excinfo.value)


def test_static_fallback_matplotlib():
    """Test if the static fallback generates a matplotlib plot."""
    optimal_weights = pd.Series({'Asset1': 0.2, 'Asset2': 0.3, 'Asset3': 0.5})
    with patch("matplotlib.pyplot.show") as mock_show:
      plot_portfolio_allocation(optimal_weights, static_fallback=True)
      mock_show.assert_called()

def test_normal_weights(mock_plotly_show):
    """Test with a normal set of portfolio weights."""
    optimal_weights = pd.Series({'Asset1': 0.2, 'Asset2': 0.3, 'Asset3': 0.5})
    plot_portfolio_allocation(optimal_weights)