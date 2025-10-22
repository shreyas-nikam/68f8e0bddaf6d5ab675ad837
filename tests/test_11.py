import pytest
import pandas as pd
from unittest.mock import MagicMock
from definition_d9df1ea236db4fd78edb2afc6532d52a import plot_efficient_frontier


def test_plot_efficient_frontier_empty_returns():
    with pytest.raises(ValueError):
        plot_efficient_frontier(pd.Series(), pd.DataFrame(), num_portfolios=10, static_fallback=False)


def test_plot_efficient_frontier_returns_covariance_mismatch():
     with pytest.raises(ValueError, match="Shape of returns and covariance matrix must match."):
        returns = pd.Series([0.1, 0.2])
        cov_matrix = pd.DataFrame([[0.01]])
        plot_efficient_frontier(returns, cov_matrix, num_portfolios=10, static_fallback=False)


def test_plot_efficient_frontier_static_fallback_matplotlib(monkeypatch):
    # Mock matplotlib to check if it's called when static_fallback is True
    mock_plt = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot", mock_plt)

    returns = pd.Series([0.1, 0.2])
    cov_matrix = pd.DataFrame([[0.01, 0.005], [0.005, 0.04]])

    plot_efficient_frontier(returns, cov_matrix, num_portfolios=10, static_fallback=True)

    assert mock_plt.figure.called
    assert mock_plt.show.called


def test_plot_efficient_frontier_plotly_called(monkeypatch):
    mock_plotly = MagicMock()
    monkeypatch.setattr("plotly.graph_objects", mock_plotly)

    returns = pd.Series([0.1, 0.2])
    cov_matrix = pd.DataFrame([[0.01, 0.005], [0.005, 0.04]])

    plot_efficient_frontier(returns, cov_matrix, num_portfolios=10, static_fallback=False)

    assert mock_plotly.Figure.called


def test_plot_efficient_frontier_insufficient_data(monkeypatch):
    mock_plotly = MagicMock()
    monkeypatch.setattr("plotly.graph_objects", mock_plotly)
    returns = pd.Series([0.1])
    cov_matrix = pd.DataFrame([[0.01]])

    plot_efficient_frontier(returns, cov_matrix, num_portfolios=10, static_fallback=False)
    assert mock_plotly.Figure.called

