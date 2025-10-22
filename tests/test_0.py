import pytest
import pandas as pd
from definition_23126b458d264c73b9fd75fe36c7c660 import generate_synthetic_financial_data

def test_generate_synthetic_financial_data_basic():
    """Test basic functionality with small number of assets and days."""
    num_assets = 2
    num_days = 10
    start_date = '2023-01-01'
    seed = 42
    df = generate_synthetic_financial_data(num_assets, num_days, start_date, seed)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == num_assets * num_days
    assert 'Date' in df.columns
    assert 'Asset_ID' in df.columns
    assert 'Open' in df.columns
    assert 'Close' in df.columns

def test_generate_synthetic_financial_data_no_assets():
    """Test with zero assets. Should return an empty DataFrame or handle gracefully."""
    num_assets = 0
    num_days = 10
    start_date = '2023-01-01'
    seed = 42
    df = generate_synthetic_financial_data(num_assets, num_days, start_date, seed)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

def test_generate_synthetic_financial_data_no_days():
    """Test with zero days. Should return an empty DataFrame or handle gracefully."""
    num_assets = 2
    num_days = 0
    start_date = '2023-01-01'
    seed = 42
    df = generate_synthetic_financial_data(num_assets, num_days, start_date, seed)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

def test_generate_synthetic_financial_data_start_date_format():
    """Test with invalid start date format. Should raise ValueError."""
    num_assets = 2
    num_days = 10
    start_date = '2023/01/01'  # Invalid format
    seed = 42

    with pytest.raises(ValueError):
        generate_synthetic_financial_data(num_assets, num_days, start_date, seed)

def test_generate_synthetic_financial_data_reproducibility():
    """Test that the seed produces the same results."""
    num_assets = 1
    num_days = 5
    start_date = '2023-01-01'
    seed = 42

    df1 = generate_synthetic_financial_data(num_assets, num_days, start_date, seed)
    df2 = generate_synthetic_financial_data(num_assets, num_days, start_date, seed)

    assert df1.equals(df2)