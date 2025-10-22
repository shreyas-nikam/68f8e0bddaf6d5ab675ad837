import pytest
import pandas as pd
from unittest.mock import MagicMock
from definition_3631a5d6eca6469cbe5561aad97c08fe import mean_variance_optimization

def test_mean_variance_optimization_basic():
    expected_returns = pd.Series([0.10, 0.15, 0.20], index=['A', 'B', 'C'])
    cov_matrix = pd.DataFrame([[0.01, 0.005, 0.002],
                               [0.005, 0.0225, 0.003],
                               [0.002, 0.003, 0.04]], index=['A', 'B', 'C'], columns=['A', 'B', 'C'])
    risk_aversion = 1.0

    # Mocking the minimize function to return a predefined result for testing purposes.
    mock_minimize_result = MagicMock()
    mock_minimize_result.x = [0.3, 0.3, 0.4]
    mock_minimize_result.success = True  # Indicate optimization was successful
    # patching the  minimize function is not possible as we don't have access to the notebook code.

    weights = mean_variance_optimization(expected_returns, cov_matrix, risk_aversion)
    assert isinstance(weights, pd.Series)
    assert len(weights) == 3
    assert (weights.index == ['A', 'B', 'C']).all()

@pytest.mark.parametrize("risk_aversion, expected_weights", [
    (0.0, [0.0, 0.0, 0.0]),
    (100.0, [0.0, 0.0, 0.0]),
])
def test_mean_variance_optimization_extreme_risk_aversion(risk_aversion, expected_weights):
    expected_returns = pd.Series([0.10, 0.15, 0.20], index=['A', 'B', 'C'])
    cov_matrix = pd.DataFrame([[0.01, 0.005, 0.002],
                               [0.005, 0.0225, 0.003],
                               [0.002, 0.003, 0.04]], index=['A', 'B', 'C'], columns=['A', 'B', 'C'])

    weights = mean_variance_optimization(expected_returns, cov_matrix, risk_aversion)
    assert isinstance(weights, pd.Series)
    assert len(weights) == 3

def test_mean_variance_optimization_zero_covariance():
    expected_returns = pd.Series([0.10, 0.15], index=['A', 'B'])
    cov_matrix = pd.DataFrame([[0.0, 0.0],
                               [0.0, 0.0]], index=['A', 'B'], columns=['A', 'B'])
    risk_aversion = 1.0

    weights = mean_variance_optimization(expected_returns, cov_matrix, risk_aversion)
    assert isinstance(weights, pd.Series)
    assert len(weights) == 2

def test_mean_variance_optimization_negative_returns():
    expected_returns = pd.Series([-0.10, -0.15], index=['A', 'B'])
    cov_matrix = pd.DataFrame([[0.01, 0.005],
                               [0.005, 0.0225]], index=['A', 'B'], columns=['A', 'B'])
    risk_aversion = 1.0

    weights = mean_variance_optimization(expected_returns, cov_matrix, risk_aversion)
    assert isinstance(weights, pd.Series)
    assert len(weights) == 2

def test_mean_variance_optimization_incorrect_input_types():
    with pytest.raises(TypeError):
        mean_variance_optimization([0.1, 0.2], [[0.1, 0.2], [0.3, 0.4]], 1.0)