import pytest
import pandas as pd
from unittest.mock import MagicMock
from definition_f53e4113aa17454e885baae0826dcbef import validate_and_summarize_data

def test_validate_and_summarize_data_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_and_summarize_data(df)

def test_validate_and_summarize_data_valid_dataframe():
    data = {'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
            'Asset_ID': [1, 2, 1, 2],
            'Open': [10.0, 20.0, 11.0, 21.0],
            'High': [12.0, 22.0, 13.0, 23.0],
            'Low': [9.0, 19.0, 10.0, 20.0],
            'Close': [11.0, 21.0, 12.0, 22.0],
            'Volume': [100, 200, 110, 210],
            'Sentiment_Score': [0.5, 0.6, 0.7, 0.8]}
    df = pd.DataFrame(data)

    # Mock the side effects of validate_and_summarize_data
    df.info = MagicMock()
    df.isnull = MagicMock(return_value=pd.DataFrame([[False] * len(df.columns)] * len(df), columns=df.columns))
    df.describe = MagicMock()

    cleaned_df = validate_and_summarize_data(df.copy())

    assert cleaned_df is not None
    assert isinstance(cleaned_df, pd.DataFrame)
    assert cleaned_df.equals(df)
    df.info.assert_called()
    df.isnull.assert_called()
    df.describe.assert_called()

def test_validate_and_summarize_data_missing_values():
    data = {'Date': ['2023-01-01', '2023-01-01'],
            'Asset_ID': [1, 2],
            'Open': [10.0, 20.0],
            'High': [12.0, 22.0],
            'Low': [9.0, 19.0],
            'Close': [11.0, 21.0],
            'Volume': [None, 200],
            'Sentiment_Score': [0.5, None]}
    df = pd.DataFrame(data)

    cleaned_df = validate_and_summarize_data(df.copy())

    assert cleaned_df['Volume'].isnull().sum() == 0
    assert cleaned_df['Sentiment_Score'].isnull().sum() == 0

def test_validate_and_summarize_data_duplicate_primary_key():
    data = {'Date': ['2023-01-01', '2023-01-01'],
            'Asset_ID': [1, 1],
            'Open': [10.0, 20.0],
            'High': [12.0, 22.0],
            'Low': [9.0, 19.0],
            'Close': [11.0, 21.0],
            'Volume': [100, 200],
            'Sentiment_Score': [0.5, 0.6]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Duplicate primary key \(Date, Asset_ID\) found"):
        validate_and_summarize_data(df)

def test_validate_and_summarize_data_incorrect_column_names():
    data = {'Datee': ['2023-01-01', '2023-01-01'],
            'Asset_ID': [1, 1],
            'Open': [10.0, 20.0],
            'High': [12.0, 22.0],
            'Low': [9.0, 19.0],
            'Close': [11.0, 21.0],
            'Volume': [100, 200],
            'Sentiment_Score': [0.5, 0.6]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="DataFrame missing expected columns"):
        validate_and_summarize_data(df)