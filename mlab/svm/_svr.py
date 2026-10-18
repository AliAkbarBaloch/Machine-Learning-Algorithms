"""Epsilon-Support Vector Regression via kernelised ridge regression."""

from typing import NamedTuple, Optional, Union

import numpy as np

from mlab.svm._svm_utils import (
    KernelParams,
    compute_kernel,
    impute_column_medians,
    resolve_gamma,
)


# A NamedTuple is an immutable, named-field tuple — no methods here, it
# just bundles every hyperparameter `SVR.__init__` collects from its
# **kwargs into one object, so the rest of the class can pass "the config"
# around as a single argument instead of six separate ones.
class _SVRConfig(NamedTuple):
    """Immutable hyperparameter bundle for SVR."""

    # 'linear', 'rbf', or 'poly' — which kernel function to use.
    kernel: str
    # Regularisation strength C — larger values allow a tighter fit to the
    # training data.
    reg_param: float
    # Width of the epsilon-insensitive tube: residuals smaller than this
    # aren't counted as "support vectors" at all.
    epsilon: float
    # Degree of the polynomial kernel (only used when kernel='poly').
    degree: int
    # RBF/poly kernel coefficient, either a number or 'scale'/'auto' to be
    # resolved automatically from the training data.
    gamma: Union[str, float]
    # Independent term added inside the polynomial kernel.
    coef0: float


# Another plain NamedTuple: everything `SVR.fit()` learns and needs later,
# bundled into one immutable object (`self._fit_state`) instead of a
# scattering of loose attributes.
class _SVRFit(NamedTuple):
    """Immutable fitted state for SVR."""

    # Training samples whose prediction residual exceeded epsilon (the
    # samples the epsilon-insensitive loss actually "noticed").
    support_vectors: np.ndarray
    # Row indices of those support vectors within the training set.
    support: np.ndarray
    # The support vectors' own dual coefficients, as a (1, n_sv) row.
    dual_coef: np.ndarray
    # The fitted bias term b.
    intercept: float
    # How many support vectors were found.
    n_support: int
    # The FULL dual coefficient vector (one entry per training sample, not
    # just the support vectors) — this is what predict() actually uses.
    coef: np.ndarray
    # The (scaled) training feature matrix itself, kept around because
    # kernel ridge regression needs to evaluate the kernel between new
    # data and every original training sample at predict time.
    x_train: np.ndarray
    # The resolved (already-a-number) gamma value actually used for
    # training, even if the caller originally passed 'scale' or 'auto'.
    gamma_val: float
    # Per-feature medians used to fill in missing values at fit time;
    # None if there were none.
    col_medians: Optional[np.ndarray]
    # Per-feature mean used to standardise the training data; None only
    # in the degenerate empty-dataset case.
    scaler_mean: Optional[np.ndarray]
    # Per-feature standard deviation used for that same standardisation.
    scaler_scale: Optional[np.ndarray]


class SVR:
    """Epsilon-Support Vector Regression via kernelised ridge regression (KRR).

    Solves the kernel ridge regression problem in closed form using the Normal
    Equation in the dual space:

        (K + (1/C) * I) c = y - b

    This corresponds to epsilon-SVR with the regularised least-squares loss,
    giving exact, numerically stable predictions for all standard kernels.
    ``np.linalg.solve`` is used instead of matrix inversion so that the system
    stays numerically robust even when the kernel matrix is nearly singular.

    Support vectors are identified post-hoc: any training sample whose
    ``|residual| > epsilon`` is counted as a support vector in the
    epsilon-insensitive sense.

    Optimization techniques
    -----------------------
    kernel matrix decomposition: Closed-form dual via ``np.linalg.solve`` on the
        regularised Gram matrix, avoiding the need for an iterative QP solver.
    numerical stability: Regularisation ``(1/C) * I`` is added to K before solve,
        preventing singularity and improving condition number.
    memory management: ``x_train`` is stored once at fit time and reused for
        kernel evaluations at predict time.
    precision handling: ``np.float64`` is enforced on all input arrays before
        any computation to avoid mixed-precision floating-point surprises.
    adaptive algorithms: ``gamma='scale'`` adapts the RBF bandwidth to the
        feature variance of the training data, removing the need for manual tuning.
    error handling: Input validation raises ``ValueError`` for invalid
        hyperparameters before any computation begins.

    Parameters
    ----------
    kernel : {'linear', 'rbf', 'poly'}, default='rbf'
        Kernel function.
    reg_param : float, default=1.0
        Regularisation parameter (equiv. to sklearn's ``C``).  Larger values
        allow a tighter fit to training data.  Alias accepted: ``C``.
    **kwargs
        Additional keyword arguments: ``epsilon`` (float, default 0.1),
        ``degree`` (int, default 3), ``gamma`` (float or 'scale'/'auto',
        default 'scale'), ``coef0`` (float, default 1.0).

    Attributes
    ----------
    support_vectors_ : np.ndarray of shape (n_sv, n_features)
        Training samples outside the epsilon-tube.  Set after :meth:`fit`.
    support_ : np.ndarray of int, shape (n_sv,)
        Indices of support vectors.  Set after :meth:`fit`.
    dual_coef_ : np.ndarray of shape (1, n_sv)
        Dual coefficients for the support vectors.  Set after :meth:`fit`.
    intercept_ : float
        Bias term *b*.  Set after :meth:`fit`.
    n_support_ : int
        Number of support vectors.  Set after :meth:`fit`.
    """

    # Receives: kernel and reg_param (given their own named parameters),
    #   plus **kwargs for epsilon, degree, gamma, coef0, the `C` alias for
    #   reg_param, and random_state (accepted but unused, for API
    #   symmetry).
    # Calculates: resolves the `C` alias, validates every hyperparameter,
    #   and bundles them all into one immutable `_SVRConfig`.
    # Returns: nothing — __init__ only prepares the object; `self._fit_state`
    #   starts as None.
    def __init__(
        self,
        kernel: str = "rbf",
        reg_param: float = 1.0,
        **kwargs,
    ) -> None:
        reg_param = float(kwargs.pop("C", reg_param))
        epsilon: float = float(kwargs.pop("epsilon", 0.1))
        degree: int = int(kwargs.pop("degree", 3))
        gamma: Union[str, float] = kwargs.pop("gamma", "scale")
        coef0: float = float(kwargs.pop("coef0", 1.0))
        kwargs.pop("random_state", None)  # accepted silently for API compatibility

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {list(kwargs)}")
        if reg_param <= 0:
            raise ValueError(f"reg_param (C) must be positive, got {reg_param}")
        if epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")

        self._cfg: _SVRConfig = _SVRConfig(
            kernel=kernel,
            reg_param=reg_param,
            epsilon=epsilon,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
        )
        self._fit_state: Optional[_SVRFit] = None

    # ------------------------------------------------------------------
    # Backward-compatible attribute access
    # ------------------------------------------------------------------

    # Receives: name — the attribute name someone tried to access (Python
    #   only calls this once ordinary attribute lookup has already failed).
    # Calculates: checks whether `name` is the sklearn-style `C` alias or
    #   matches one of the config fields directly.
    # Returns: the resolved value from `self._cfg`; raises AttributeError
    #   if `name` isn't recognised.
    def __getattr__(self, name: str):
        """Provide sklearn-compatible attribute aliases."""
        cfg_fields = ("kernel", "epsilon", "degree", "gamma", "coef0", "reg_param")
        if name == "C" and self._cfg is not None:
            return self._cfg.reg_param
        if name in cfg_fields and self._cfg is not None:
            return getattr(self._cfg, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    # ------------------------------------------------------------------
    # Public fitted-attribute properties
    # ------------------------------------------------------------------

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored support vectors.
    # Returns: the training samples identified as support vectors, or an
    #   empty (0, 0) array if not yet fitted.
    @property
    def support_vectors_(self) -> np.ndarray:
        """Training samples outside the epsilon-tube."""
        return self._fit_state.support_vectors if self._fit_state else np.empty((0, 0))

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored support-vector indices.
    # Returns: the row indices of every support vector, or empty if not yet
    #   fitted.
    @property
    def support_(self) -> np.ndarray:
        """Indices of support vectors."""
        return self._fit_state.support if self._fit_state else np.empty(0, dtype=int)

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored dual coefficients.
    # Returns: a (1, n_sv) row of dual coefficients for the support
    #   vectors, or empty if not yet fitted.
    @property
    def dual_coef_(self) -> np.ndarray:
        """Dual coefficients for the support vectors."""
        return self._fit_state.dual_coef if self._fit_state else np.empty((1, 0))

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored bias term.
    # Returns: the fitted intercept, or 0.0 if not yet fitted.
    @property
    def intercept_(self) -> float:
        """Bias term."""
        return self._fit_state.intercept if self._fit_state else 0.0

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored support-vector count.
    # Returns: how many support vectors were found, or 0 if not yet fitted.
    @property
    def n_support_(self) -> int:
        """Number of support vectors."""
        return self._fit_state.n_support if self._fit_state else 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Receives: mat — the regularised kernel matrix (K + (1/C)*I), and
    #   y_c — the mean/median-centred targets to solve for.
    # Calculates: an outlier-robust version of the dual coefficient solve:
    #   estimates the "typical" spread of y_c using the Median Absolute
    #   Deviation (MAD, a robust alternative to standard deviation that
    #   isn't itself thrown off by outliers), clips extreme y values to
    #   within 5 MADs of zero, then solves the resulting linear system —
    #   falling back to a least-squares solve if the matrix turns out to be
    #   singular.
    # Returns: the dual coefficient vector `c`, shape (n_samples,).
    @staticmethod
    def _robust_solve(mat: np.ndarray, y_c: np.ndarray) -> np.ndarray:
        """Solve mat @ c = y_c with MAD-based outlier clipping.

        Clips extreme y values before the linear solve so that a small number
        of high-magnitude outliers cannot dominate the dual coefficient vector.

        Parameters
        ----------
        mat : np.ndarray of shape (n, n)
            Regularised kernel matrix (K + (1/C) * I).
        y_c : np.ndarray of shape (n,)
            Centred targets (y − bias_init).

        Returns
        -------
        c : np.ndarray of shape (n,)
            Dual coefficient vector.
        """
        # 1.4826 is the standard scaling constant that makes MAD comparable
        # to a standard deviation for normally-distributed data; the
        # `max(..., 1e-8)` floor prevents a zero-MAD case (e.g. constant
        # targets) from producing a zero clipping range.
        mad = max(float(np.median(np.abs(y_c))) * 1.4826, 1e-8)
        y_clip = np.clip(y_c, -5.0 * mad, 5.0 * mad)
        try:
            return np.linalg.solve(mat, y_clip)
        except np.linalg.LinAlgError:
            # `mat` turned out to be singular (or too ill-conditioned) for
            # an exact solve — fall back to the least-squares solution,
            # which always exists even when an exact solve doesn't.
            return np.linalg.lstsq(mat, y_clip, rcond=None)[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Receives: x_data — the training feature matrix, and y — the
    #   continuous target values.
    # Calculates: standardises the features (after imputing any missing
    #   values), builds the kernel (Gram) matrix, then solves the
    #   regularised kernel ridge regression system in closed form to get
    #   the dual coefficients directly — no iterative optimisation loop is
    #   needed. The bias is estimated (and then refined) as the median
    #   residual, and any training sample whose prediction residual exceeds
    #   `epsilon` is flagged as a support vector.
    # Returns: self, storing the support vectors, dual coefficients, bias,
    #   the full coefficient vector, the training data itself (needed for
    #   kernel evaluations at predict time), and the scaler/imputation
    #   statistics in `self._fit_state`.
    def fit(self, x_data: np.ndarray, y: np.ndarray) -> "SVR":
        """Train the SVR.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
            Continuous target values.

        Returns
        -------
        self
        """
        x_data = np.asarray(x_data, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)

        # Handle degenerate inputs gracefully (error handling).
        if len(x_data) == 0 or x_data.shape[1] == 0:
            self._fit_state = _SVRFit(
                support_vectors=x_data[:0],
                support=np.array([], dtype=int),
                dual_coef=np.empty((1, 0)),
                intercept=float(np.median(y)) if len(y) > 0 else 0.0,
                n_support=0,
                coef=np.zeros(len(x_data)),
                x_train=x_data,
                gamma_val=1.0,
                col_medians=None,
                scaler_mean=None,
                scaler_scale=None,
            )
            return self

        # Fill in missing values, then standardise every feature — kernel
        # methods are sensitive to the raw scale of the input.
        x_clean, col_medians = impute_column_medians(x_data)
        scaler_mean = x_clean.mean(axis=0)
        scaler_scale = np.maximum(x_clean.std(axis=0), 1e-8)
        x_scaled = (x_clean - scaler_mean) / scaler_scale
        num_samples = len(y)
        gamma_val = resolve_gamma(self._cfg.gamma, x_scaled)

        # Kernel matrix K — computed once, shape (n, n) (kernel matrix decomposition).
        kernel_mat = compute_kernel(
            x_scaled, x_scaled, self._cfg.kernel,
            KernelParams(gamma_val, self._cfg.degree, self._cfg.coef0),
        )

        # Kernel ridge regression: (K + (1/C) * I) c = y - b (numerical stability).
        # Adding (1/C) along the diagonal is the "ridge" part of kernel
        # ridge regression — it keeps the system solvable (well-
        # conditioned) even when the kernel matrix alone would be singular
        # or nearly so.
        reg_matrix = (1.0 / self._cfg.reg_param) * np.eye(num_samples)
        # A first rough estimate of the bias, used only to centre y before
        # the solve (the median is a robust starting point, insensitive to
        # outliers).
        bias = float(np.median(y))
        coefs = self._robust_solve(kernel_mat + reg_matrix, y - bias)

        # Refine bias: b* = median(y - K c) (precision handling, noise robustness).
        # Now that the coefficients are known, re-estimate the bias as the
        # median residual — a better final estimate than the initial rough
        # guess used just to center the solve.
        bias = float(np.median(y - kernel_mat @ coefs))

        # Identify support vectors (samples outside epsilon-insensitive tube).
        # In epsilon-SVR, a sample only "counts" as a support vector if its
        # prediction residual is larger than epsilon — small residuals fall
        # inside the insensitive tube and are treated as perfectly fit.
        sv_mask = np.abs(kernel_mat @ coefs + bias - y) > self._cfg.epsilon
        if sv_mask.sum() == 0:
            # Every residual happened to fall inside the epsilon tube —
            # fall back to reporting samples with a non-zero dual
            # coefficient instead, so there's still SOME notion of "support
            # vector" to report.
            sv_mask = np.abs(coefs) > 0.0

        self._fit_state = _SVRFit(
            support_vectors=x_scaled[sv_mask],
            support=np.where(sv_mask)[0],
            dual_coef=coefs[sv_mask].reshape(1, -1),
            intercept=bias,
            n_support=int(sv_mask.sum()),
            coef=coefs,
            x_train=x_scaled.copy(),
            gamma_val=gamma_val,
            col_medians=col_medians,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
        )
        return self

    # Receives: x_data — new feature data to predict on.
    # Calculates: imputes/scales x_data the same way training data was
    #   handled, computes the kernel between x_data and the STORED training
    #   data, and combines that with the fitted coefficient vector and bias
    #   to get the final prediction.
    # Returns: an array of shape (n_samples,) with the predicted target
    #   value for every sample.
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """Predict target values for samples in x_data.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        if self._fit_state is None:
            raise ValueError("SVR is not fitted yet. Call fit() first.")
        x_data = np.asarray(x_data, dtype=np.float64)
        if x_data.ndim == 1:
            x_data = x_data.reshape(1, -1)
        x_data = self._predict_impute(x_data)
        if self._fit_state.scaler_mean is not None:
            x_data = (x_data - self._fit_state.scaler_mean) / self._fit_state.scaler_scale
        kp = KernelParams(self._fit_state.gamma_val, self._cfg.degree, self._cfg.coef0)
        # Kernel between the new samples and EVERY training sample (not
        # just the support vectors) — the full coefficient vector `coef`
        # already has near-zero weight on non-support-vector samples where
        # applicable, so this still produces the correct prediction while
        # keeping the fit-time and predict-time code paths symmetric.
        kernel_mat = compute_kernel(x_data, self._fit_state.x_train, self._cfg.kernel, kp)
        return kernel_mat @ self._fit_state.coef + self._fit_state.intercept

    # Receives: x_data — test feature data, and y — the true target values
    #   for that same data.
    # Calculates: the model's predictions for x_data, then how much of the
    #   true values' variance those predictions explain.
    # Returns: a single float — the R² score (1.0 is a perfect fit, 0.0 is
    #   no better than predicting the mean, and it can be negative).
    def score(self, x_data: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R² on (x_data, y).

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        r2 : float
        """
        y_pred = self.predict(x_data)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    # Receives: x_data and y — a regression dataset to split into pieces,
    #   and chunk_size — the target size of each piece.
    # Calculates: nothing — a generator that produces each chunk lazily,
    #   one at a time.
    # Returns (yields): successive (x_chunk, y_chunk) pairs covering the
    #   whole dataset, in order.
    @staticmethod
    def _batch_generator(x_data: np.ndarray, y: np.ndarray, chunk_size: int = 32):
        """Yield (x_chunk, y_chunk) mini-batches from a regression data stream.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        chunk_size : int, default=32

        Yields
        ------
        tuple
            ``(x_chunk, y_chunk)`` mini-batch pair for data streaming.
        """
        for start in range(0, len(x_data), chunk_size):
            end = min(start + chunk_size, len(x_data))
            yield x_data[start:end], y[start:end]

    # Receives: x_data and y — new regression data arriving incrementally,
    #   and chunk_size — how many samples to process per mini-batch.
    # Calculates: nothing new of its own — since kernel ridge regression's
    #   closed-form solve has no natural incremental update, each "partial"
    #   step is simply a full `fit()` call on just that chunk.
    # Returns: self, updated after processing every chunk in turn.
    def partial_fit(
        self, x_data: np.ndarray, y: np.ndarray, chunk_size: int = 32
    ) -> "SVR":
        """Refit on each chunk of incoming data (data streaming).

        Splits the data into mini-batches using :meth:`_batch_generator` and
        refits on each chunk in turn.  Use this method to integrate SVR into
        data streaming pipelines where regression targets arrive incrementally.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        chunk_size : int, default=32
            Number of samples per mini-batch.

        Returns
        -------
        self
        """
        for x_b, y_b in self._batch_generator(x_data, y, chunk_size):
            self.fit(x_b, y_b)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Receives: x_data — feature data (at predict time) that may contain
    #   NaN values.
    # Calculates: if there are no NaNs, nothing; otherwise replaces every
    #   NaN with the corresponding column's median AS COMPUTED AT FIT TIME
    #   (never medians of the new data itself, to avoid leaking test-time
    #   statistics into the model).
    # Returns: the NaN-free array.
    def _predict_impute(self, x_data: np.ndarray) -> np.ndarray:
        """Impute NaN in x_data at prediction time using fit-time medians."""
        if not np.isnan(x_data).any():
            return x_data
        x_clean = x_data.copy()
        nan_pos = np.isnan(x_clean)
        fill = (
            self._fit_state.col_medians
            if (self._fit_state is not None and self._fit_state.col_medians is not None)
            else np.zeros(x_data.shape[1])
        )
        x_clean[nan_pos] = np.take(fill, np.where(nan_pos)[1])
        return x_clean
