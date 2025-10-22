import pytest
from definition_b2b5b22216e84be3be4c99f570022b3d import split_and_scale_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

@pytest.fixture
def sample_dataframe():
    data = {'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10],
            'target': [11, 12, 13, 14, 15],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])}
    df = pd.DataFrame(data)
    return df

def test_split_and_scale_data_shapes(sample_dataframe):
    train_ratio = 0.8
    features = ['feature1', 'feature2']
    target = 'target'
    X_train, X_test, y_train, y_test, _ = split_and_scale_data(sample_dataframe, train_ratio, features, target)
    assert X_train.shape[0] == 4
    assert X_test.shape[0] == 1
    assert y_train.shape[0] == 4
    assert y_test.shape[0] == 1
    assert X_train.shape[1] == 2
    assert X_test.shape[1] == 2

def test_split_and_scale_data_scaling(sample_dataframe):
    train_ratio = 0.8
    features = ['feature1', 'feature2']
    target = 'target'
    X_train, X_test, _, _, scaler = split_and_scale_data(sample_dataframe, train_ratio, features, target)
    assert isinstance(scaler, StandardScaler)
    assert np.allclose(X_train.mean(axis=0), 0)  # Check if training data is scaled (approximately)
    assert not np.allclose(X_test.mean(axis=0), 0) #Check if test data is scaled

def test_split_and_scale_data_chronological_split(sample_dataframe):
    train_ratio = 0.8
    features = ['feature1', 'feature2']
    target = 'target'
    X_train, X_test, _, _, _ = split_and_scale_data(sample_dataframe, train_ratio, features, target)
    assert sample_dataframe['date'].iloc[0] == X_train['date'].iloc[0]
    assert sample_dataframe['date'].iloc[-1] == X_test['date'].iloc[0]

def test_split_and_scale_data_empty_features(sample_dataframe):
    train_ratio = 0.8
    features = []
    target = 'target'
    with pytest.raises(ValueError):  # Expect ValueError if no features are provided.
        split_and_scale_data(sample_dataframe, train_ratio, features, target)

def test_split_and_scale_data_invalid_train_ratio(sample_dataframe):
    train_ratio = -0.1
    features = ['feature1', 'feature2']
    target = 'target'
    with pytest.raises(ValueError):  # Expect ValueError if train_ratio is not between 0 and 1
        split_and_scale_data(sample_dataframe, train_ratio, features, target)