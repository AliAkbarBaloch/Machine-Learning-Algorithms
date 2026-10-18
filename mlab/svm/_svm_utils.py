"""Shared kernel functions and gamma-resolution helpers for the mlab SVM module."""

from typing import NamedTuple, Tuple, Union

import numpy as np


# A NamedTuple is a lightweight, immutable "struct" — it behaves like a
# regular tuple (can be unpacked, compared, etc.) but its fields also have
# names, so `params.gamma` reads far better than `params[0]`. There are no
# methods here on purpose: this class exists purely to carry three related
# numbers around as one argument instead of three separate ones (see the
# docstring below — pylint would otherwise flag `compute_kernel` for taking
# too many parameters).
class KernelParams(NamedTuple):
    """Kernel hyperparameters bundled to stay within pylint's max-args limit.

    Attributes
    ----------
    gamma : float
        Kernel coefficient for RBF and polynomial kernels.
    degree : int
        Degree for the polynomial kernel.
    coef0 : float
        Independent term for the polynomial kernel.
    """

    # Kernel coefficient — how strongly the RBF kernel penalises distance,
    # or how the polynomial kernel scales the dot product before raising
    # it to a power.
    gamma: float
    # How many times the polynomial kernel's (gamma * dot + coef0) term is
    # multiplied by itself.
    degree: int
    # A constant added inside the polynomial kernel before exponentiation
    # (lets the kernel represent lower-degree terms too, not just the
    # pure degree-`degree` term).
    coef0: float


# Receives: x_data — the raw feature matrix (possibly 1-D, if there's only
#   one feature), and y — the raw target vector (class labels or
#   regression targets).
# Calculates: converts x_data into a proper 2-D float64 array (reshaping a
#   flat 1-D input into a single column), converts y into a plain array,
#   and finds every distinct value y contains.
# Returns: a 3-tuple (x_2d, y_arr, classes) — the cleaned-up feature
#   matrix, the cleaned-up target array, and the sorted array of unique
#   target values.
def prepare_fit_inputs(
    x_data: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast and reshape raw fit inputs into standard 2-D form.

    Parameters
    ----------
    x_data : np.ndarray
        Raw feature matrix; may be 1-D (single-feature case).
    y : np.ndarray
        Target vector (class labels or regression targets).

    Returns
    -------
    x_2d : np.ndarray of shape (n_samples, n_features)
        Float64 array, guaranteed 2-D.
    y_arr : np.ndarray of shape (n_samples,)
        Target vector cast to ndarray.
    classes : np.ndarray
        Sorted unique values of y_arr.
    """
    x_data = np.asarray(x_data, dtype=np.float64)
    y = np.asarray(y)
    if x_data.ndim == 1:
        # A single-feature dataset given as a flat (n,) array gets turned
        # into an (n, 1) column, so every downstream function can always
        # assume a proper 2-D feature matrix.
        x_data = x_data.reshape(-1, 1)
    return x_data, y, np.unique(y)


# Receives: x_arr1 and x_arr2 — two sets of samples to compare (they can be
#   the same set, e.g. when building a training Gram matrix), kernel — which
#   kernel function to use ('linear', 'rbf', or 'poly'), and params — the
#   bundled gamma/degree/coef0 hyperparameters.
# Calculates: the kernel value between every sample in x_arr1 and every
#   sample in x_arr2, all at once via vectorised matrix operations.
# Returns: an array of shape (n1, n2) — the kernel (Gram) matrix, where
#   entry [i, j] is k(x_arr1[i], x_arr2[j]).
def compute_kernel(
    x_arr1: np.ndarray,
    x_arr2: np.ndarray,
    kernel: str = "linear",
    params: KernelParams = KernelParams(gamma=1.0, degree=3, coef0=1.0),
) -> np.ndarray:
    """Evaluate the kernel matrix kernel_mat[i,j] = k(x_arr1[i], x_arr2[j]).

    Parameters
    ----------
    x_arr1 : np.ndarray of shape (n1, n_features)
        First set of samples.
    x_arr2 : np.ndarray of shape (n2, n_features)
        Second set of samples.
    kernel : {'linear', 'rbf', 'poly'}
        Kernel function type.
    params : KernelParams
        Bundled hyperparameters: ``gamma``, ``degree``, ``coef0``.

    Returns
    -------
    kernel_mat : np.ndarray of shape (n1, n2)
        Kernel (Gram) matrix.
    """
    if kernel == "linear":
        # The linear kernel is just the plain dot product between every
        # pair of samples — one matrix multiply computes all of them at
        # once.
        return x_arr1 @ x_arr2.T

    if kernel == "rbf":
        # ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj
        # Expanding the squared distance this way avoids ever forming an
        # explicit (n1, n2, n_features) array of per-feature differences —
        # everything here is a plain 2-D matrix operation.
        sq1 = np.sum(x_arr1 ** 2, axis=1, keepdims=True)   # (n1, 1)
        sq2 = np.sum(x_arr2 ** 2, axis=1)                   # (n2,)
        # `sq1 + sq2` broadcasts an (n1, 1) column against an (n2,) row,
        # producing every (i, j) pair's ||xi||^2 + ||xj||^2 sum at once;
        # subtracting `2 * x_arr1 @ x_arr2.T` (the pairwise dot products)
        # completes the expanded squared-distance formula for every pair.
        sq_dist = sq1 + sq2 - 2.0 * (x_arr1 @ x_arr2.T)
        # Floating-point rounding can occasionally push a "should be
        # exactly 0" distance slightly negative; clamp it back to 0 before
        # the kernel's exponential, so we never end up with exp(positive)
        # blowing up where exp(0) was intended.
        np.maximum(sq_dist, 0.0, out=sq_dist)
        return np.exp(-params.gamma * sq_dist)

    if kernel == "poly":
        # Polynomial kernel: (gamma * <x_i, x_j> + coef0) ^ degree.
        return (params.gamma * (x_arr1 @ x_arr2.T) + params.coef0) ** params.degree

    raise ValueError(
        f"Unknown kernel '{kernel}'. Supported kernels: 'linear', 'rbf', 'poly'."
    )


# Receives: x_data — a feature matrix that may contain NaN values.
# Calculates: if there are no NaNs at all, does nothing; otherwise computes
#   each column's median (ignoring NaNs) and fills every NaN entry with its
#   own column's median (columns that are entirely NaN fall back to 0.0).
# Returns: a tuple (x_clean, col_medians) — the NaN-free array (or the
#   original array unchanged if there were no NaNs), and the per-column
#   medians used (or None if nothing needed imputing) so the caller can
#   reuse the exact same medians later at predict time.
def impute_column_medians(x_data: np.ndarray):
    """Replace NaN values with per-column medians, returning both the cleaned array
    and the medians used so callers can apply the same transform at predict time.

    Columns that are entirely NaN are filled with 0.

    Parameters
    ----------
    x_data : np.ndarray of shape (n_samples, n_features)
        Input array, may contain NaN.

    Returns
    -------
    x_clean : np.ndarray of shape (n_samples, n_features)
        Copy of x_data with NaN replaced by column medians (or x_data itself
        when no NaN are present).
    col_medians : np.ndarray of shape (n_features,) or None
        Per-column medians used for imputation.  ``None`` when x_data had no NaN.
    """
    if not np.isnan(x_data).any():
        # Nothing to impute — return the array unchanged (no need to
        # allocate a copy).
        return x_data, None
    # `np.isnan(x_data).all(axis=0)` marks any column that is ENTIRELY
    # NaN (nanmedian on such a column would itself be NaN); those columns
    # get 0.0 instead, every other column gets its own real median.
    col_medians = np.where(
        np.isnan(x_data).all(axis=0), 0.0, np.nanmedian(x_data, axis=0)
    )
    x_clean = x_data.copy()
    nan_pos = np.isnan(x_clean)
    # `np.where(nan_pos)[1]` gives the COLUMN index of every NaN entry;
    # `np.take(col_medians, ...)` looks up that column's own median for
    # each one, so every NaN gets replaced by the median of the column it
    # actually belongs to.
    x_clean[nan_pos] = np.take(col_medians, np.where(nan_pos)[1])
    return x_clean, col_medians


# Receives: gamma — either an explicit numeric value, or one of the
#   strings 'scale'/'auto' asking to compute gamma automatically from the
#   data; and x_data — the training data (only used when gamma is a
#   string).
# Calculates: if gamma is already a number, nothing; if it's 'scale',
#   1 / (n_features * Var[X]); if it's 'auto', 1 / n_features.
# Returns: a single float — the concrete gamma value to actually use.
def resolve_gamma(gamma: Union[str, float], x_data: np.ndarray) -> float:
    """Resolve gamma='scale' / 'auto' to a float value.

    Parameters
    ----------
    gamma : float or {'scale', 'auto'}
        Kernel coefficient specification.
    x_data : np.ndarray of shape (n_samples, n_features)
        Training data (used only when gamma is a string).

    Returns
    -------
    gamma_val : float
        Resolved numeric gamma value.
    """
    if isinstance(gamma, str):
        if gamma == "scale":
            # sklearn's gamma='scale' formula: 1 / (n_features * overall
            # variance of X). Falls back to 1.0 if the data has literally
            # zero variance (e.g. every feature is constant), to avoid
            # dividing by zero.
            var = x_data.var()
            return 1.0 / (x_data.shape[1] * var) if var > 0 else 1.0
        if gamma == "auto":
            return 1.0 / x_data.shape[1]
        raise ValueError(
            f"Unknown gamma string '{gamma}'. Use 'scale', 'auto', or a float."
        )
    return float(gamma)
