import pytest
import pandas as pd
from definition_1dbc3be7a0f8461096b6c04c35709007 import engineer_features

@pytest.fixture
def sample_dataframe():
    data = {'Asset_ID': [1, 1, 2, 2],
            'Close': [10, 11, 20, 19]}
    return pd.DataFrame(data)

def test_engineer_features_empty_dataframe():
    df = pd.DataFrame()
    result = engineer_features(df, lags=[1], rolling_window_ma=5, rolling_window_vol=5)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_engineer_features_no_close_column(sample_dataframe):
    df = sample_dataframe.drop('Close', axis=1)
    with pytest.raises(KeyError):
        engineer_features(df, lags=[1], rolling_window_ma=5, rolling_window_vol=5)

def test_engineer_features_basic_functionality(sample_dataframe):
    result = engineer_features(sample_dataframe.copy(), lags=[1], rolling_window_ma=2, rolling_window_vol=2)
    assert isinstance(result, pd.DataFrame)
    assert 'daily_return' in result.columns
    assert 'lagged_return_1' in result.columns
    assert 'moving_average_20' in result.columns
    assert 'rolling_volatility_20' in result.columns

def test_engineer_features_with_different_lags(sample_dataframe):
    result = engineer_features(sample_dataframe.copy(), lags=[1, 3], rolling_window_ma=2, rolling_window_vol=2)
    assert 'lagged_return_1' in result.columns
    assert 'lagged_return_3' in result.columns

def test_engineer_features_rolling_window_zero(sample_dataframe):

    result = engineer_features(sample_dataframe.copy(), lags=[1], rolling_window_ma=0, rolling_window_vol=0)
    assert isinstance(result, pd.DataFrame)
    assert 'moving_average_20' in result.columns
    assert 'rolling_volatility_20' in result.columns
