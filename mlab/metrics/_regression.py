"""Regression scoring metrics shared across every regressor in this library
(LinearRegressor, SGDRegression, SVR, MLPRegressor).

`LinearRegressor`/`SGDRegression`/`SVR` each already compute their own R2
inline (`r2score_` / `score()`), returning 1.0 for a constant target —
that's left untouched. `r2_score` here follows scikit-learn's convention
instead (see its docstring). This module exists so new code, and
anything that wants MAE/RMSE/MAPE or wants to compare several models on the
same footing, has one shared, tested place to get them from instead of
every notebook re-deriving the formulas by hand.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

from mlab.metrics._shared import check_same_length, rank


# Receives: `y` — a list or array of numbers, possibly not float64 yet and
#   possibly shaped as a column vector instead of a flat list.
# Calculates: converts `y` into a numpy array of 64-bit floats, and flattens
#   it to 1-D if it came in as a 2-D column vector.
# Returns: a flat (n,) float64 numpy array.
def _as_float_array(y: Union[Sequence, NDArray]) -> NDArray:
    # np.asarray turns a plain Python list (or an existing array) into a
    # numpy array of 64-bit floats. If it's already a float64 array this is
    # basically free (no copy); if it's a list or an int array, this is what
    # lets every line below use fast vectorised math instead of Python loops.
    arr = np.asarray(y, dtype=np.float64)
    # Some callers might pass in a column vector of shape (n, 1) instead of
    # a flat (n,) array (e.g. straight out of a model that returns 2D
    # output). `.ravel()` flattens it to 1D so the rest of this file can
    # always assume a simple (n,) shape.
    return arr.ravel() if arr.ndim > 1 else arr


# Receives: y_true and y_pred — the true and predicted target values.
# Calculates: the average of the squared difference between each true value
#   and its corresponding prediction.
# Returns: a single float, the Mean Squared Error (always >= 0).
def mean_squared_error(
    y_true: Union[Sequence, NDArray], y_pred: Union[Sequence, NDArray]
) -> float:
    """Average squared residual. Penalises big misses much harder than small
    ones because the errors are squared before averaging.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    float, >= 0
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    check_same_length(y_true, y_pred, "regression")
    # `y_true - y_pred` subtracts element-by-element (this is what "vectorised"
    # means: no explicit for-loop, numpy does it for every element at once).
    # `** 2` squares each of those differences, and `np.mean` averages them
    # into a single number. This is literally MSE = average((true - pred)^2).
    return float(np.mean((y_true - y_pred) ** 2))


# Receives: y_true and y_pred — the true and predicted target values.
# Calculates: the mean squared error (by calling `mean_squared_error`), then
#   its square root.
# Returns: a single float, the Root Mean Squared Error (always >= 0), in the
#   same units as the target itself.
def root_mean_squared_error(
    y_true: Union[Sequence, NDArray], y_pred: Union[Sequence, NDArray]
) -> float:
    """Square root of `mean_squared_error`, back in the same units as the
    target — usually the number people actually want to report.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    float, >= 0
    """
    # RMSE is just MSE with a square root taken at the very end, so we reuse
    # mean_squared_error instead of recomputing the sum of squared errors.
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# Receives: y_true and y_pred — the true and predicted target values.
# Calculates: the average of the absolute (unsquared) difference between
#   each true value and its corresponding prediction.
# Returns: a single float, the Mean Absolute Error (always >= 0).
def mean_absolute_error(
    y_true: Union[Sequence, NDArray], y_pred: Union[Sequence, NDArray]
) -> float:
    """Average absolute residual — unlike MSE, a single bad outlier doesn't
    dominate the whole score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    float, >= 0
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    check_same_length(y_true, y_pred, "regression")
    # np.abs strips the sign off each (true - pred) difference so a
    # prediction that's "5 too high" and one that's "5 too low" count the
    # same, then np.mean averages all those absolute errors together.
    return float(np.mean(np.abs(y_true - y_pred)))


# Receives: y_true and y_pred, plus `eps` — a tiny floor value used to avoid
#   dividing by exactly zero.
# Calculates: for every sample, the absolute error expressed as a fraction
#   of that sample's true value (with `eps` protecting against a true value
#   of exactly 0), then averages those fractions across all samples.
# Returns: a single float, the Mean Absolute Percentage Error, expressed as
#   a plain fraction (e.g. 0.05 means "5% off on average"), always >= 0.
def mean_absolute_percentage_error(
    y_true: Union[Sequence, NDArray],
    y_pred: Union[Sequence, NDArray],
    eps: float = 1e-12,
) -> float:
    """Average absolute error expressed as a fraction of the true value —
    handy when the target spans several orders of magnitude and a raw MAE
    number wouldn't mean much on its own.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)
    eps : floor applied to `abs(y_true)` before dividing, so a handful of
        exactly-zero targets don't blow the whole average up to infinity.

    Returns
    -------
    float, >= 0 (reported as a fraction, not multiplied by 100)
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    check_same_length(y_true, y_pred, "regression")
    # np.maximum compares two arrays (or an array and a scalar) element by
    # element and keeps whichever is bigger at each position. Here it makes
    # sure we never divide by a number smaller than `eps`, even if some
    # entries of y_true are exactly 0 — that would otherwise produce a
    # division-by-zero (an infinite percentage error) for that one sample.
    denom = np.maximum(np.abs(y_true), eps)
    # For every sample: how big is the error relative to the true value?
    # Then average all of those per-sample percentage errors into one number.
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


# Receives: y_true and y_pred — the true and predicted target values.
# Calculates: how much smaller the model's total squared error is than the
#   total squared error of a trivial "always predict the average" baseline,
#   expressed as a fraction of that baseline's error.
# Returns: a single float — 1.0 for a perfect fit, 0.0 for "no better than
#   guessing the mean", and possibly negative for a model worse than that.
def r2_score(y_true: Union[Sequence, NDArray], y_pred: Union[Sequence, NDArray]) -> float:
    """Coefficient of determination: the fraction of the target's variance
    that the predictions explain. 1.0 is a perfect fit, 0.0 is no better
    than always predicting the mean, and it can go negative if the model
    is worse than that.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    float
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    check_same_length(y_true, y_pred, "regression")

    # "Residual" just means "leftover error" — how far off each prediction
    # was from the truth. Squaring and summing those errors gives one
    # number representing the model's total mistake (SS_res = sum of
    # squared residuals). Smaller is better; 0 means a perfect fit.
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    # The baseline we're comparing against is "a model that always predicts
    # the average of y_true". `np.mean(y_true)` computes that average, and
    # SS_tot measures how spread out the true values are around it — i.e.
    # how much error that trivial baseline model would rack up.
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot == 0.0:
        # every true value is identical, so "variance explained" isn't really
        # defined — the only sane answer is 1.0 if the fit is exact and 0.0
        # otherwise, rather than dividing by zero
        return 1.0 if ss_res == 0.0 else 0.0
    # R^2 = 1 - (our model's error) / (the baseline's error). If our model
    # makes zero error, this is 1 - 0 = 1.0 (perfect). If our model does
    # exactly as badly as always-predict-the-mean, this is 1 - 1 = 0.0.
    return 1.0 - ss_res / ss_tot


# Receives: y_true and y_pred — the true and predicted target values.
# Calculates: how strongly the two series move together in a straight-line
#   (linear) relationship, independent of their scale or units.
# Returns: a single float between -1 and 1 — +1 means a perfect increasing
#   straight-line relationship, -1 a perfect decreasing one, 0 means no
#   linear relationship at all.
def pearson_correlation(
    y_true: Union[Sequence, NDArray], y_pred: Union[Sequence, NDArray]
) -> float:
    """Linear correlation between the true and predicted values — how much
    of the relationship is captured by a straight line, regardless of
    scale or offset (unlike R², a model that gets the trend right but is
    off by a constant factor still scores well here).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    float in [-1, 1]
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    check_same_length(y_true, y_pred, "regression")

    # "Centering" a series means subtracting its own mean from every entry,
    # so the result has an average of 0. This removes the effect of scale
    # (e.g. one series being measured in meters, the other in centimeters)
    # and lets us focus purely on how the two series move together.
    true_centered = y_true - y_true.mean()
    pred_centered = y_pred - y_pred.mean()
    # The Pearson formula is: sum(x_centered * y_centered) divided by
    # sqrt(sum(x_centered^2) * sum(y_centered^2)). The denominator here is
    # exactly that square-root term — it normalises the result so the final
    # answer always falls between -1 and 1, no matter the original units.
    denom = np.sqrt(np.sum(true_centered ** 2) * np.sum(pred_centered ** 2))
    if denom == 0.0:
        # one of the two series is constant, so there's no linear
        # relationship to speak of — 0 rather than a 0/0 NaN
        return 0.0
    # The numerator, sum(true_centered * pred_centered), is large and
    # positive when both series tend to be above their own mean (or both
    # below it) at the same time — i.e. when they move together.
    return float(np.sum(true_centered * pred_centered) / denom)


# Receives: y_true and y_pred — the true and predicted target values.
# Calculates: converts both series to ranks (1st, 2nd, 3rd, ... smallest to
#   largest) and then computes the ordinary Pearson correlation on those
#   ranks instead of on the raw numbers.
# Returns: a single float between -1 and 1 — rewards any consistently
#   increasing or decreasing relationship, even a curved (non-linear) one.
def spearman_correlation(
    y_true: Union[Sequence, NDArray], y_pred: Union[Sequence, NDArray]
) -> float:
    """Pearson correlation computed on the *ranks* of the values instead of
    the raw numbers. Where `pearson_correlation` only rewards a straight-line
    relationship, this rewards any consistently increasing (or decreasing)
    one — useful when a model gets the ordering right but the exact curve
    isn't linear.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    float in [-1, 1]
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    check_same_length(y_true, y_pred, "regression")
    # `rank(...)` replaces every value with "what position would this be in
    # if the array were sorted?" (e.g. the smallest value becomes 1, the
    # next becomes 2, and so on — see `_shared.rank` for the full
    # walkthrough, including how tied values are handled). Once both arrays
    # are turned into ranks, running the ordinary Pearson formula on those
    # ranks is the definition of Spearman correlation.
    return pearson_correlation(rank(y_true), rank(y_pred))


# Receives: y_true and y_pred — the true and predicted target values.
# Calculates: how much of the spread (variance) in the true values is
#   mirrored by the predictions, after ignoring any constant offset between
#   them (a residual that's always "+5" doesn't count against this metric).
# Returns: a single float, <= 1.0 — 1.0 is a perfect fit (up to a constant
#   offset).
def explained_variance_score(
    y_true: Union[Sequence, NDArray], y_pred: Union[Sequence, NDArray]
) -> float:
    """Fraction of the target's variance explained by the predictions,
    ignoring any constant offset between them. Close to `r2_score` — the
    difference only shows up when predictions are systematically biased
    (e.g. consistently 5 units too high): R² would penalise that bias,
    this metric wouldn't.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    float, <= 1.0
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    check_same_length(y_true, y_pred, "regression")

    # np.var computes the variance of an array: how spread out its values
    # are around their own mean. `true_variance` is "how much natural spread
    # is there in the target we're trying to predict?"
    true_variance = float(np.var(y_true))
    # `y_true - y_pred` is the array of per-sample errors (the residuals).
    # Its variance tells us how spread out the *errors themselves* are —
    # note this is different from r2_score's SS_res, which sums squared
    # errors rather than measuring their spread around the error's own mean.
    # That's precisely what makes this metric ignore a constant offset: a
    # residual array that's always "+5" has zero variance (no spread) even
    # though it's not zero.
    residual_variance = float(np.var(y_true - y_pred))
    if true_variance == 0.0:
        # nothing to explain — perfect only if the residual is also constant
        return 1.0 if residual_variance == 0.0 else 0.0
    # How much of the target's natural spread is "left over" as error
    # spread, subtracted from 1. If the residual has no spread at all
    # relative to the target, this is 1 - 0 = 1.0 (perfect).
    return 1.0 - residual_variance / true_variance


# Receives: y_true and y_pred — the true and predicted target values — and
#   `digits`, how many decimal places to print each number with.
# Calculates: every regression metric defined above, once each, on the same
#   (y_true, y_pred) pair.
# Returns: a single multi-line string — a print-ready table listing MSE,
#   RMSE, MAE, MAPE, R², Explained Variance, Pearson r and Spearman r.
def regression_report(
    y_true: Union[Sequence, NDArray], y_pred: Union[Sequence, NDArray], digits: int = 4
) -> str:
    """Print-ready summary of every regression metric above, computed once
    on the same (y_true, y_pred) pair.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)
    digits : how many decimal places to print.

    Returns
    -------
    str
    """
    # Just call every metric function above once each, and remember the
    # results in plain variables so we can format them into a table next.
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson = pearson_correlation(y_true, y_pred)
    spearman = spearman_correlation(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    # Build the report one line at a time. `{'MSE':>14}` means "print the
    # text MSE right-aligned inside a 14-character-wide column", and
    # `{mse:.{digits}f}` means "print the number `mse` as a fixed-point
    # decimal with `digits` digits after the decimal point". Lining every
    # row up this way is what makes the printed table look like a table.
    rows = [
        f"{'MSE':>14}  {mse:.{digits}f}",
        f"{'RMSE':>14}  {rmse:.{digits}f}",
        f"{'MAE':>14}  {mae:.{digits}f}",
        f"{'MAPE':>14}  {mape:.{digits}f}",
        f"{'R^2':>14}  {r2:.{digits}f}",
        f"{'Explained Var':>14}  {evs:.{digits}f}",
        f"{'Pearson r':>14}  {pearson:.{digits}f}",
        f"{'Spearman r':>14}  {spearman:.{digits}f}",
    ]
    # "\n".join(rows) glues all the lines together with a newline character
    # between each one, turning the list of strings into one big string
    # that prints as a multi-line table.
    return "\n".join(rows)
