import pytest
import pandas as pd
from definition_e5095dadcd5f4d44a5c8a7a225c3282e import engineer_features

@pytest.fixture
def sample_dataframe():
    data = {'Asset_ID': [1, 1, 2, 2],
            'Close': [100, 102, 50, 51]}
    return pd.DataFrame(data)

def test_engineer_features_empty_df():
    df = pd.DataFrame()
    result = engineer_features(df, lags=[1], rolling_window_ma=5, rolling_window_vol=5)
    assert result.empty

def test_engineer_features_lags(sample_dataframe):
    df = sample_dataframe.copy()
    lags = [1, 2]
    result = engineer_features(df, lags=lags, rolling_window_ma=5, rolling_window_vol=5)
    for lag in lags:
        assert f'Close_Lag_{lag}' in result.columns

def test_engineer_features_rolling_ma_and_vol(sample_dataframe):
    df = sample_dataframe.copy()
    rolling_window_ma = 2
    rolling_window_vol = 2
    result = engineer_features(df, lags=[1], rolling_window_ma=rolling_window_ma, rolling_window_vol=rolling_window_vol)
    assert 'Close_MA_2' in result.columns
    assert 'Volatility_2' in result.columns

def test_engineer_features_no_asset_id(sample_dataframe):
    df = sample_dataframe.copy()
    del df['Asset_ID']
    with pytest.raises(KeyError):
         engineer_features(df, lags=[1], rolling_window_ma=5, rolling_window_vol=5)

def test_engineer_features_defaults(sample_dataframe):
    df = sample_dataframe.copy()
    result = engineer_features(df, lags=[1,5,20], rolling_window_ma=20, rolling_window_vol=20)
    assert 'Daily_Return' in result.columns
    assert 'Close_Lag_1' in result.columns
    assert 'Close_MA_20' in result.columns
    assert 'Volatility_20' in result.columns
