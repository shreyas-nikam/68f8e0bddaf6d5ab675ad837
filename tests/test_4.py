import pytest
from definition_608d4ff11994433c8cd3336641d2a9cc import split_and_scale_data
import pandas as pd
from sklearn.preprocessing import StandardScaler

@pytest.fixture
def sample_dataframe():
    data = {'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10],
            'target': [11, 12, 13, 14, 15],
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])}
    return pd.DataFrame(data)

def test_split_and_scale_data_valid(sample_dataframe):
    train_ratio = 0.8
    features = ['feature1', 'feature2']
    target = 'target'
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(sample_dataframe, train_ratio, features, target)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert isinstance(scaler, StandardScaler)
    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1

def test_split_and_scale_data_empty_dataframe():
    df = pd.DataFrame()
    train_ratio = 0.8
    features = ['feature1', 'feature2']
    target = 'target'
    with pytest.raises(ValueError):
        split_and_scale_data(df, train_ratio, features, target)

def test_split_and_scale_data_invalid_train_ratio(sample_dataframe):
    train_ratio = 1.2
    features = ['feature1', 'feature2']
    target = 'target'
    with pytest.raises(ValueError):
         split_and_scale_data(sample_dataframe, train_ratio, features, target)

def test_split_and_scale_data_no_features(sample_dataframe):
    train_ratio = 0.8
    features = []
    target = 'target'
    with pytest.raises(ValueError):
        split_and_scale_data(sample_dataframe, train_ratio, features, target)

def test_split_and_scale_data_missing_target(sample_dataframe):
    train_ratio = 0.8
    features = ['feature1', 'feature2']
    target = 'missing_target'
    with pytest.raises(KeyError):
        split_and_scale_data(sample_dataframe, train_ratio, features, target)