"""Linear and logistic regression (closed-form, batch GD and mini-batch SGD)."""

from mlab.regression._linear import LinearRegressor, SGDRegression
from mlab.regression._logistic import LogisticRegression, SGDClassifier

__all__ = ["LinearRegressor", "SGDRegression", "LogisticRegression", "SGDClassifier"]
