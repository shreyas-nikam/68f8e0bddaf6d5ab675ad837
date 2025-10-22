import pytest
import pandas as pd
from definition_f5c2f63d7cf64fd5a5abf12bd1d37809 import define_target_variable

def test_define_target_variable_empty_df():
    df = pd.DataFrame({'Close': [], 'Asset_ID': []})
    with pytest.raises(Exception):
        define_target_variable(df, forward_days=1)

def test_define_target_variable_single_asset():
    data = {'Close': [10, 11, 12, 13, 14],
            'Asset_ID': ['A'] * 5,
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])}
    df = pd.DataFrame(data)
    df = df.set_index(['Date','Asset_ID'])
    result_df = define_target_variable(df.copy(), forward_days=1)
    expected_return = (11 - 10) / 10
    assert 'Future_Return' in result_df.columns
    assert abs(result_df['Future_Return'].iloc[0] - expected_return) < 1e-6

def test_define_target_variable_multiple_assets():
    data = {'Close': [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
            'Asset_ID': ['A'] * 5 + ['B'] * 5,
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'] * 2)}
    df = pd.DataFrame(data)
    df = df.set_index(['Date','Asset_ID'])
    result_df = define_target_variable(df.copy(), forward_days=1)
    assert 'Future_Return' in result_df.columns
    assert len(result_df) == 8

def test_define_target_variable_forward_days_greater_than_data_length():
    data = {'Close': [10, 11, 12],
            'Asset_ID': ['A'] * 3,
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])}
    df = pd.DataFrame(data)
    df = df.set_index(['Date','Asset_ID'])
    result_df = define_target_variable(df.copy(), forward_days=5)
    assert len(result_df) == 0

def test_define_target_variable_zero_forward_days():
    data = {'Close': [10, 11, 12],
            'Asset_ID': ['A'] * 3,
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])}
    df = pd.DataFrame(data)
    df = df.set_index(['Date','Asset_ID'])
    result_df = define_target_variable(df.copy(), forward_days=0)

    expected_return_1 = (10-10)/10
    expected_return_2 = (11-11)/11
    expected_return_3 = (12-12)/12
    assert abs(result_df['Future_Return'].iloc[0] - expected_return_1) < 1e-6
    assert abs(result_df['Future_Return'].iloc[1] - expected_return_2) < 1e-6
    assert abs(result_df['Future_Return'].iloc[2] - expected_return_3) < 1e-6
