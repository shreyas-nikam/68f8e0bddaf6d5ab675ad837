import pytest
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.exceptions import NotFittedError
from definition_da47259382a34b1f85ea9df403d42f8b import train_predictive_model


def test_train_predictive_model_basic():
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([7, 8, 9])
    alpha_regularization = 1.0
    model = train_predictive_model(X_train, y_train, alpha_regularization)
    assert isinstance(model, Ridge)
    assert model.alpha == alpha_regularization


def test_train_predictive_model_empty_input():
    X_train = pd.DataFrame()
    y_train = pd.Series()
    alpha_regularization = 1.0
    with pytest.raises(ValueError):
        train_predictive_model(X_train, y_train, alpha_regularization)


def test_train_predictive_model_alpha_zero():
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([7, 8, 9])
    alpha_regularization = 0.0
    model = train_predictive_model(X_train, y_train, alpha_regularization)
    assert isinstance(model, Ridge)
    assert model.alpha == alpha_regularization


def test_train_predictive_model_alpha_large():
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([7, 8, 9])
    alpha_regularization = 1000.0
    model = train_predictive_model(X_train, y_train, alpha_regularization)
    assert isinstance(model, Ridge)
    assert model.alpha == alpha_regularization

def test_train_predictive_model_different_index():
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}, index=[0,1,2])
    y_train = pd.Series([7, 8, 9], index=[3,4,5])
    alpha_regularization = 1.0
    with pytest.raises(ValueError): # or maybe it should train it anyway and discard some data... depends on expected behaviour
        train_predictive_model(X_train, y_train, alpha_regularization)