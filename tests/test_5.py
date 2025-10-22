import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from definition_b78179a542194559ab6601441e49bd61 import train_predictive_model

@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [6, 7, 8, 9, 10]})
    y_train = pd.Series([11, 12, 13, 14, 15])
    return X_train, y_train

def test_train_predictive_model_basic(sample_data):
    X_train, y_train = sample_data
    alpha = 1.0
    model = train_predictive_model(X_train, y_train, alpha)
    assert isinstance(model, Ridge)
    assert model.alpha == alpha

def test_train_predictive_model_alpha_zero(sample_data):
    X_train, y_train = sample_data
    alpha = 0.0
    model = train_predictive_model(X_train, y_train, alpha)
    assert isinstance(model, Ridge)
    assert model.alpha == alpha

def test_train_predictive_model_large_alpha(sample_data):
    X_train, y_train = sample_data
    alpha = 100.0
    model = train_predictive_model(X_train, y_train, alpha)
    assert isinstance(model, Ridge)
    assert model.alpha == alpha

def test_train_predictive_model_empty_data():
    X_train = pd.DataFrame()
    y_train = pd.Series()
    alpha = 1.0
    with pytest.raises(Exception):
        train_predictive_model(X_train, y_train, alpha)

def test_train_predictive_model_invalid_alpha(sample_data):
    X_train, y_train = sample_data
    alpha = -1.0
    with pytest.raises(ValueError):
        train_predictive_model(X_train, y_train, alpha)