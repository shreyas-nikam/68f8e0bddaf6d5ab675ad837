import pytest
from definition_4a27807b31f745e3a988d2c5644717b7 import validate_and_summarize_data
import pandas as pd
import numpy as np

def create_sample_dataframe(include_nan=False):
    data = {'Date': ['2023-01-01', '2023-01-02', '2023-01-01'],
            'Asset_ID': [1, 2, 1],
            'Open': [10.0, 20.0, 10.0],
            'High': [12.0, 22.0, 11.0],
            'Low': [9.0, 19.0, 9.5],
            'Close': [11.0, 21.0, 10.5],
            'Volume': [100, 200, 150],
            'Sentiment_Score': [0.5, 0.7, 0.6]}

    if include_nan:
        data['Volume'][0] = np.nan
        data['Sentiment_Score'][1] = np.nan

    df = pd.DataFrame(data)
    return df

def test_validate_and_summarize_data_valid_df():
    df = create_sample_dataframe()
    cleaned_df = validate_and_summarize_data(df.copy())
    assert cleaned_df is not None
    assert cleaned_df.equals(df)

def test_validate_and_summarize_data_missing_values_imputation():
    df = create_sample_dataframe(include_nan=True)
    cleaned_df = validate_and_summarize_data(df.copy())

    assert not cleaned_df['Volume'].isnull().any()
    assert not cleaned_df['Sentiment_Score'].isnull().any()

def test_validate_and_summarize_data_duplicate_primary_key():
    df = create_sample_dataframe()

    with pytest.raises(Exception) as excinfo:
      validate_and_summarize_data(df.copy())

    assert "DataFrame does not have unique primary key" in str(excinfo.value)

def test_validate_and_summarize_data_empty_df():
    df = pd.DataFrame()
    with pytest.raises(Exception) as excinfo:
        validate_and_summarize_data(df.copy())
    assert "DataFrame is empty." in str(excinfo.value)

def test_validate_and_summarize_data_incorrect_column_names():
    df = create_sample_dataframe()
    df = df.rename(columns={'Date': 'WrongDate'})
    with pytest.raises(Exception) as excinfo:
      validate_and_summarize_data(df.copy())
    assert "DataFrame does not contain the required columns" in str(excinfo.value)