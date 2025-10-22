import pytest
import pandas as pd
from definition_5169ec4498404e62a57ae212d8e0b28f import estimate_portfolio_parameters

def test_estimate_portfolio_parameters_empty_df():
    df_returns = pd.DataFrame()
    predictions = pd.Series()
    with pytest.raises(ValueError):
        estimate_portfolio_parameters(df_returns, predictions)

def test_estimate_portfolio_parameters_mismatched_assets():
    df_returns = pd.DataFrame({'A': [0.01, 0.02], 'B': [0.03, 0.04]})
    predictions = pd.Series([0.05, 0.06, 0.07], index=['A', 'B', 'C'])
    expected_returns, cov_matrix = estimate_portfolio_parameters(df_returns, predictions)
    assert set(expected_returns.index) == set(['A', 'B'])
    assert set(cov_matrix.index) == set(['A', 'B'])

def test_estimate_portfolio_parameters_valid_input():
    df_returns = pd.DataFrame({'A': [0.01, 0.02], 'B': [0.03, 0.04]})
    predictions = pd.Series([0.05, 0.06], index=['A', 'B'])
    expected_returns, cov_matrix = estimate_portfolio_parameters(df_returns, predictions)
    assert isinstance(expected_returns, pd.Series)
    assert isinstance(cov_matrix, pd.DataFrame)
    assert set(expected_returns.index) == set(['A', 'B'])
    assert set(cov_matrix.index) == set(['A', 'B'])
    assert cov_matrix.shape == (2, 2)

def test_estimate_portfolio_parameters_single_asset():
    df_returns = pd.DataFrame({'A': [0.01, 0.02]})
    predictions = pd.Series([0.05], index=['A'])
    expected_returns, cov_matrix = estimate_portfolio_parameters(df_returns, predictions)
    assert isinstance(expected_returns, pd.Series)
    assert isinstance(cov_matrix, pd.DataFrame)
    assert expected_returns.index[0] == 'A'
    assert cov_matrix.shape == (1, 1)

def test_estimate_portfolio_parameters_different_index_types():
    df_returns = pd.DataFrame({'A': [0.01, 0.02], 'B': [0.03, 0.04]}, index=[0,1])
    predictions = pd.Series([0.05, 0.06], index=['A', 'B'])
    with pytest.raises(TypeError):
       estimate_portfolio_parameters(df_returns, predictions)