import pytest
import pandas as pd
from unittest.mock import MagicMock
from definition_5abc8a1fa9a54079b2340db8aa82986f import estimate_portfolio_parameters


def test_estimate_portfolio_parameters_typical_case():
    df_returns = pd.DataFrame({'Asset1': [0.01, 0.02, -0.01], 'Asset2': [0.02, -0.01, 0.01]})
    predictions = pd.Series({'Asset1': 0.03, 'Asset2': 0.04})

    expected_returns, cov_matrix = estimate_portfolio_parameters(df_returns, predictions)

    assert isinstance(expected_returns, pd.Series)
    assert isinstance(cov_matrix, pd.DataFrame)
    assert list(expected_returns.index) == ['Asset1', 'Asset2']
    assert list(cov_matrix.columns) == ['Asset1', 'Asset2']


def test_estimate_portfolio_parameters_empty_dataframe():
    df_returns = pd.DataFrame()
    predictions = pd.Series()

    with pytest.raises(ValueError):
        estimate_portfolio_parameters(df_returns, predictions)



def test_estimate_portfolio_parameters_mismatched_assets():
    df_returns = pd.DataFrame({'Asset1': [0.01, 0.02], 'Asset2': [0.02, 0.03]})
    predictions = pd.Series({'Asset3': 0.03, 'Asset4': 0.04})
    with pytest.raises(ValueError):
        estimate_portfolio_parameters(df_returns, predictions)


def test_estimate_portfolio_parameters_single_asset():
    df_returns = pd.DataFrame({'Asset1': [0.01, 0.02, -0.01]})
    predictions = pd.Series({'Asset1': 0.03})

    expected_returns, cov_matrix = estimate_portfolio_parameters(df_returns, predictions)

    assert isinstance(expected_returns, pd.Series)
    assert isinstance(cov_matrix, pd.DataFrame)
    assert expected_returns.index[0] == 'Asset1'
    assert cov_matrix.columns[0] == 'Asset1'

def test_estimate_portfolio_parameters_predictions_are_none():
    df_returns = pd.DataFrame({'Asset1': [0.01, 0.02, -0.01], 'Asset2': [0.02, -0.01, 0.01]})
    predictions = None

    with pytest.raises(TypeError):
        estimate_portfolio_parameters(df_returns, predictions)