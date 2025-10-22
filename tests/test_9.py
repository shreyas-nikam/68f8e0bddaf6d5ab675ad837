import pytest
import pandas as pd
from unittest.mock import MagicMock
from definition_a4ec73d05dde4dbcb551f5f697312011 import mean_variance_optimization


def test_mean_variance_optimization_basic():
    expected_returns = pd.Series([0.1, 0.2, 0.15])
    cov_matrix = pd.DataFrame([[0.01, 0.005, 0.002],
                               [0.005, 0.04, 0.003],
                               [0.002, 0.003, 0.02]])
    risk_aversion = 1.0
    weights = mean_variance_optimization(expected_returns, cov_matrix, risk_aversion)
    assert isinstance(weights, pd.Series)
    assert len(weights) == len(expected_returns)
    assert (weights >= 0).all()


def test_mean_variance_optimization_zero_risk_aversion():
    expected_returns = pd.Series([0.1, 0.2, 0.15])
    cov_matrix = pd.DataFrame([[0.01, 0.005, 0.002],
                               [0.005, 0.04, 0.003],
                               [0.002, 0.003, 0.02]])
    risk_aversion = 0.0  # No risk aversion
    weights = mean_variance_optimization(expected_returns, cov_matrix, risk_aversion)
    assert isinstance(weights, pd.Series)
    assert len(weights) == len(expected_returns)


def test_mean_variance_optimization_high_risk_aversion():
    expected_returns = pd.Series([0.1, 0.2, 0.15])
    cov_matrix = pd.DataFrame([[0.01, 0.005, 0.002],
                               [0.005, 0.04, 0.003],
                               [0.002, 0.003, 0.02]])
    risk_aversion = 100.0  # Very high risk aversion
    weights = mean_variance_optimization(expected_returns, cov_matrix, risk_aversion)
    assert isinstance(weights, pd.Series)
    assert len(weights) == len(expected_returns)


def test_mean_variance_optimization_negative_expected_returns():
     expected_returns = pd.Series([-0.1, 0.2, -0.15])
     cov_matrix = pd.DataFrame([[0.01, 0.005, 0.002],
                                [0.005, 0.04, 0.003],
                                [0.002, 0.003, 0.02]])
     risk_aversion = 1.0
     weights = mean_variance_optimization(expected_returns, cov_matrix, risk_aversion)
     assert isinstance(weights, pd.Series)
     assert len(weights) == len(expected_returns)
     assert (weights >= 0).all()

def test_mean_variance_optimization_invalid_input():
    expected_returns = [0.1, 0.2, 0.15] #not pd.Series
    cov_matrix = pd.DataFrame([[0.01, 0.005, 0.002],
                               [0.005, 0.04, 0.003],
                               [0.002, 0.003, 0.02]])
    risk_aversion = 1.0
    with pytest.raises(TypeError):
        mean_variance_optimization(expected_returns, cov_matrix, risk_aversion)