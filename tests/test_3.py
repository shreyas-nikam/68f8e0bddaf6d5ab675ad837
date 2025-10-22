import pytest
import pandas as pd
from definition_55cbed8a0d884a368fef6d9df169e08d import define_target_variable


def create_sample_df(data):
    return pd.DataFrame(data)


def test_define_target_variable_basic():
    data = {'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Asset_ID': [1, 1, 1],
            'Close': [100, 101, 102]}
    df = create_sample_df(data)
    result = define_target_variable(df.copy(), forward_days=1)
    assert 'Future_Return' in result.columns
    assert len(result) == 2
    assert result['Future_Return'].iloc[0] == (101 - 100) / 100
    assert result['Future_Return'].iloc[1] == (102 - 101) / 101


def test_define_target_variable_multiple_assets():
    data = {'Date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
            'Asset_ID': [1, 1, 2, 2],
            'Close': [100, 101, 50, 51]}
    df = create_sample_df(data)
    result = define_target_variable(df.copy(), forward_days=1)
    assert len(result) == 2


def test_define_target_variable_forward_days_2():
        data = {'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
            'Asset_ID': [1, 1, 1, 1],
            'Close': [100, 101, 102, 103]}
        df = create_sample_df(data)
        result = define_target_variable(df.copy(), forward_days=2)
        assert 'Future_Return' in result.columns
        assert len(result) == 2
        assert result['Future_Return'].iloc[0] == (102 - 100) / 100
        assert result['Future_Return'].iloc[1] == (103 - 101) / 101

def test_define_target_variable_empty_dataframe():
    data = {'Date': [], 'Asset_ID': [], 'Close': []}
    df = create_sample_df(data)
    result = define_target_variable(df.copy(), forward_days=1)
    assert len(result) == 0

def test_define_target_variable_zero_close_price():
    data = {'Date': ['2023-01-01', '2023-01-02'],
            'Asset_ID': [1, 1],
            'Close': [0, 10]}
    df = create_sample_df(data)
    result = define_target_variable(df.copy(), forward_days=1)
    assert 'Future_Return' in result.columns
    assert len(result) == 0
