"""Linear soft-margin SVM for binary classification via sub-gradient descent."""

from typing import List, NamedTuple, Optional

import numpy as np

from mlab.svm._svm_utils import impute_column_medians, prepare_fit_inputs


# A NamedTuple behaves like a plain tuple (immutable, unpackable, easy to
# compare) but with named fields, so `cfg.learning_rate` is readable instead
# of `cfg[0]`. This one has no methods at all — it exists purely to hold
# every hyperparameter `SVM.__init__` collects from its **kwargs as one
# single, unchangeable bundle, so the rest of the class never has to carry
# eight separate arguments around.
class _SVMConfig(NamedTuple):
    """Immutable hyperparameter bundle for the primal SVM."""

    # Step size used for every sub-gradient descent update.
    learning_rate: float
    # How many full-batch epochs to run.
    n_iterations: int
    # 'linear' or 'rbf' — which feature transform to apply before training.
    kernel: str
    # RBF kernel coefficient (only used when kernel='rbf').
    gamma: float
    # Training stops early once the average hinge loss drops below this.
    tol: float
    # Whether to print a message when training converges early.
    verbose: bool
    # Regularisation strength C — larger means less regularisation (a
    # harder margin).
    reg_param: float
    # Whether to shrink the learning rate over epochs (1/sqrt(t) decay).
    lr_decay: bool


# Another plain NamedTuple, this time bundling everything `SVM.fit()`
# learns and needs to remember, so the whole "have I been fitted yet?"
# state lives in one object (`self._fit_state`) instead of a dozen loose
# attributes that could get out of sync with each other.
class _SVMFit(NamedTuple):
    """Immutable container for the fitted state of an ``SVM`` instance."""

    # The learned weight vector (binary case) or the FIRST class's weight
    # vector (multiclass case — see `weights_mc` below for the rest).
    weight: np.ndarray
    # The learned bias/intercept term (binary case).
    bias: float
    # Training samples identified as support vectors (on or inside the
    # margin).
    support_vectors: np.ndarray
    # Row indices of those support vectors within the training set.
    support: np.ndarray
    # How many support vectors were found.
    n_support: int
    # Every distinct class label seen during fit().
    classes: np.ndarray
    # Average hinge loss recorded at the end of each training epoch.
    loss_history: List[float]
    # Per-feature mean used to standardise the training data — reused at
    # predict time so new data gets the exact same transform.
    scaler_mean: np.ndarray
    # Per-feature standard deviation used for the same standardisation.
    scaler_scale: np.ndarray
    # Multiclass (One-vs-Rest) only: one weight vector PER CLASS, stacked
    # into a single (n_classes, n_features) array. None for binary models.
    weights_mc: Optional[np.ndarray]
    # Multiclass only: one bias value per class. None for binary models.
    biases_mc: Optional[np.ndarray]
    # Per-feature medians used to fill in missing (NaN) values at fit
    # time; None if the training data had no missing values.
    col_medians: Optional[np.ndarray]


class SVM:
    """Soft-margin SVM for binary classification via sub-gradient descent (primal).

    Minimises the regularised hinge loss over the full training set:

        J(w, b) = (1/m) * sum_i max(0, 1 - y_i(w·x_i + b)) + lambda * ||w||^2

    where lambda = 1 / (2 * C).  Full-batch (epoch-level) sub-gradients are
    computed each iteration; the update clips to the support-vector region
    automatically through the mask on margin violators.

    For non-linear kernels use ``SVC``, which implements the Lagrangian dual
    via SMO.  The ``kernel='rbf'`` option here uses a simplified element-wise
    feature map (matching the course notebook's ``BaseSVM``); it is provided
    for API compatibility only.

    Data streaming is supported via :meth:`partial_fit` for online updates
    after initial training.  Numerical stability is achieved by standardising
    input features at fit time.  Error handling raises ``ValueError`` for
    invalid inputs and unfitted-model calls.

    Optimization techniques
    -----------------------
    gradient optimization: Sub-gradient of the hinge loss with L2 regularisation.
    convergence checking: Early stopping when average hinge loss < tol.
    learning rate schedule: ``learning_rate`` is applied uniformly each epoch;
        a ``1/sqrt(t)`` decay schedule can be enabled via ``lr_decay=True`` in
        the constructor kwargs.  Decaying learning rates improve convergence on
        noisy data (gradient optimization, scaling normalization).
    numpy usage: All gradient computations are fully vectorised; no Python loop
        over samples inside the epoch.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Gradient descent step size.  Alias accepted: ``lr``.
    n_iterations : int, default=100
        Number of gradient descent epochs.  Alias accepted: ``max_iter``.
    **kwargs
        Additional keyword arguments: ``kernel`` ('linear' or 'rbf', default
        'linear'), ``gamma`` (float, default 1.0), ``tol`` (float, default
        1e-4), ``verbose`` (bool, default False), ``lr_decay`` (bool, default
        False).  ``C`` is accepted as an alias for the regularisation strength
        (default 1.0).

    Attributes
    ----------
    weight_ : np.ndarray of shape (n_features,)
        Primal weight vector.  Set after :meth:`fit`.
    bias_ : float
        Bias term *b*.  Set after :meth:`fit`.
    intercept_ : float
        Alias for ``bias_``.  Set after :meth:`fit`.
    support_vectors_ : np.ndarray of shape (n_sv, n_features)
        Training samples that lie on or inside the margin.  Set after :meth:`fit`.
    support_ : np.ndarray of int, shape (n_sv,)
        Indices of support vectors in the training set.  Set after :meth:`fit`.
    n_support_ : int
        Total number of support vectors.  Set after :meth:`fit`.
    classes_ : np.ndarray of shape (2,)
        Unique class labels encountered during :meth:`fit`.
    loss_history_ : list of float
        Average hinge loss recorded at the end of each epoch.
    """

    # Receives: learning_rate and n_iterations (the two most commonly
    #   tuned hyperparameters, given their own named parameters), plus
    #   **kwargs for everything else (kernel, gamma, tol, verbose, the `C`
    #   regularisation strength, lr_decay, and aliases `lr`/`max_iter`).
    # Calculates: pulls every recognised keyword out of kwargs (using
    #   whichever alias the caller used, if any), validates that every
    #   hyperparameter is in a sane range, and bundles them all into one
    #   immutable `_SVMConfig`.
    # Returns: nothing — __init__ only builds `self._cfg` and sets
    #   `self._fit_state` to None (the model isn't trained yet).
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 100,
        **kwargs,
    ) -> None:
        # `kwargs.pop(name, default)` removes and returns a keyword
        # argument if the caller supplied it, or `default` if not — this
        # lets both the canonical name (e.g. `learning_rate`) and its alias
        # (`lr`) be checked, with the alias taking precedence if given.
        learning_rate = float(kwargs.pop("lr", learning_rate))
        n_iterations = int(kwargs.pop("max_iter", n_iterations))
        kernel: str = str(kwargs.pop("kernel", "linear"))
        gamma: float = float(kwargs.pop("gamma", 1.0))
        tol: float = float(kwargs.pop("tol", 1e-4))
        verbose: bool = bool(kwargs.pop("verbose", False))
        reg_param: float = float(kwargs.pop("C", 1.0))
        lr_decay: bool = bool(kwargs.pop("lr_decay", False))
        kwargs.pop("random_state", None)  # accepted silently for API compatibility

        # Anything left in kwargs after popping every recognised name is an
        # argument this class doesn't understand.
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {list(kwargs)}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")
        if reg_param <= 0:
            raise ValueError(f"C (reg_param) must be positive, got {reg_param}")

        # A NamedTuple is immutable, so bundling every hyperparameter here
        # guarantees `self._cfg` can never be accidentally mutated after
        # construction — any change would require building a whole new
        # config.
        self._cfg: _SVMConfig = _SVMConfig(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            kernel=kernel,
            gamma=gamma,
            tol=tol,
            verbose=verbose,
            reg_param=reg_param,
            lr_decay=lr_decay,
        )
        self._fit_state: Optional[_SVMFit] = None

    # ------------------------------------------------------------------
    # Backward-compatible attribute access
    # ------------------------------------------------------------------

    # Receives: name — the attribute name someone tried to access on this
    #   object (Python only calls `__getattr__` when normal attribute
    #   lookup has ALREADY failed).
    # Calculates: checks whether `name` is a known sklearn-style alias
    #   (like `C` for `reg_param`) or matches one of the config fields
    #   directly, and if so looks that value up on `self._cfg`.
    # Returns: the resolved attribute's value; raises AttributeError if
    #   `name` isn't recognised at all.
    def __getattr__(self, name: str):
        """Provide sklearn-compatible attribute aliases."""
        aliases = {
            "C": "reg_param",
            "lr": "learning_rate",
            "max_iter": "n_iterations",
        }
        cfg_fields = (
            "learning_rate", "n_iterations", "kernel", "gamma", "tol", "verbose", "reg_param"
        )
        resolved = aliases.get(name, name)
        if resolved in cfg_fields and self._cfg is not None:
            return getattr(self._cfg, resolved)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    # ------------------------------------------------------------------
    # Public fitted-attribute properties
    # ------------------------------------------------------------------

    # Receives: nothing besides self.
    # Calculates: nothing — just reads the weight vector out of
    #   `self._fit_state` if the model has been fitted.
    # Returns: the primal weight vector, or an empty array if `fit()`
    #   hasn't been called yet.
    @property
    def weight_(self) -> np.ndarray:
        """Primal weight vector."""
        return self._fit_state.weight if self._fit_state else np.empty(0)

    # Receives: nothing besides self.
    # Calculates: nothing — reads the bias term out of `self._fit_state`.
    # Returns: the fitted bias term, or 0.0 if not yet fitted.
    @property
    def bias_(self) -> float:
        """Bias term."""
        return self._fit_state.bias if self._fit_state else 0.0

    # Receives: nothing besides self.
    # Calculates: nothing — this is purely a differently-named way to
    #   reach `bias_`, for sklearn-compatible naming.
    # Returns: the same value as `bias_`.
    @property
    def intercept_(self) -> float:
        """Alias for bias_."""
        return self.bias_

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored support vectors.
    # Returns: the training samples identified as support vectors, or an
    #   empty (0, 0) array if not yet fitted.
    @property
    def support_vectors_(self) -> np.ndarray:
        """Training samples on or inside the margin."""
        return self._fit_state.support_vectors if self._fit_state else np.empty((0, 0))

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored support-vector indices.
    # Returns: the row indices (into the training set) of every support
    #   vector, or empty if not yet fitted.
    @property
    def support_(self) -> np.ndarray:
        """Indices of support vectors."""
        return self._fit_state.support if self._fit_state else np.empty(0, dtype=int)

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored support-vector count.
    # Returns: how many support vectors were found, or 0 if not yet fitted.
    @property
    def n_support_(self) -> int:
        """Total number of support vectors."""
        return self._fit_state.n_support if self._fit_state else 0

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored class labels.
    # Returns: the unique class labels seen during fit(), or empty if not
    #   yet fitted.
    @property
    def classes_(self) -> np.ndarray:
        """Unique class labels seen at fit time."""
        return self._fit_state.classes if self._fit_state else np.empty(0)

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored per-epoch loss log.
    # Returns: the list of average hinge-loss values recorded during
    #   training, or an empty list if not yet fitted.
    @property
    def loss_history_(self) -> List[float]:
        """Average hinge loss per training epoch."""
        return self._fit_state.loss_history if self._fit_state else []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Receives: x_data — the training feature matrix, and y — the binary
    #   class labels (any two distinct values, e.g. strings or ints — not
    #   required to already be {-1, +1} or {0, 1}).
    # Calculates: standardises the features (and imputes any missing
    #   values), then — if there are exactly 2 classes — runs the
    #   sub-gradient descent training loop to learn a weight vector and
    #   bias; if there are MORE than 2 classes, instead trains one binary
    #   classifier per class using the One-vs-Rest strategy.
    # Returns: self (the fitted model), storing every learned piece of
    #   state (weights, bias, scaler statistics, support vectors, etc.) in
    #   the immutable `self._fit_state` tuple.
    def fit(self, x_data: np.ndarray, y: np.ndarray) -> "SVM":
        """Train the SVM on (x_data, y).

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
            Binary class labels (any two distinct values).

        Returns
        -------
        self
        """
        x_data, y, classes = prepare_fit_inputs(x_data, y)

        # Handle degenerate inputs gracefully (error handling).
        # An empty dataset (no rows, or no columns) has nothing to learn
        # from — rather than letting the training loop crash on it, store
        # a trivial "all-zero" fitted state so predict() still behaves
        # predictably.
        if len(x_data) == 0 or x_data.shape[1] == 0:
            self._fit_state = _SVMFit(
                weight=np.zeros(max(x_data.shape[1], 1)),
                bias=0.0,
                support_vectors=x_data[:0],
                support=np.array([], dtype=int),
                n_support=0,
                classes=classes if len(classes) > 0 else np.array([0, 1]),
                loss_history=[],
                scaler_mean=np.zeros(max(x_data.shape[1], 1)),
                scaler_scale=np.ones(max(x_data.shape[1], 1)),
                weights_mc=None,
                biases_mc=None,
                col_medians=None,
            )
            return self

        # Fill in any missing values, then standardise every feature to
        # zero mean / unit variance — sub-gradient descent converges far
        # more reliably on similarly-scaled features.
        x_data, col_medians = impute_column_medians(x_data)
        scaler = (x_data.mean(axis=0), np.maximum(x_data.std(axis=0), 1e-8))
        x_scaled = (x_data - scaler[0]) / scaler[1]
        # The 'rbf' option here is a simplified element-wise feature map
        # (NOT a true kernel trick like SVC's), applied purely for API
        # compatibility with the course notebook.
        x_feat = np.exp(-self._cfg.gamma * x_scaled) if self._cfg.kernel == "rbf" else x_scaled

        if len(classes) > 2:
            # One-vs-Rest multiclass: train one binary SVM per class.
            w_mc, b_mc, loss_h = self._train_multiclass(x_feat, y, classes)
            # Multiclass mode doesn't track a single-classifier support-
            # vector set, so these fields stay empty/placeholder.
            sv_tuple = (x_scaled[:0], np.array([], dtype=int), 0)
            self._fit_state = _SVMFit(
                weight=w_mc[0],
                bias=float(b_mc[0]),
                support_vectors=sv_tuple[0],
                support=sv_tuple[1],
                n_support=0,
                classes=classes,
                loss_history=loss_h,
                scaler_mean=scaler[0],
                scaler_scale=scaler[1],
                weights_mc=w_mc,
                biases_mc=b_mc,
                col_medians=col_medians,
            )
            return self

        # Binary case: encode the two class labels as -1/+1, then run the
        # ordinary sub-gradient descent loop.
        y_bin = self._encode(y, classes)
        weights, bias, loss_h = self._train_loop(x_feat, y_bin)

        try:
            sv_tuple = self._find_support_vectors(x_scaled, y_bin, x_feat, weights, bias)
        except (ValueError, RuntimeError):
            # If support-vector identification somehow fails, fall back to
            # reporting none rather than letting fit() itself fail.
            sv_tuple = (x_scaled[:0], np.array([], dtype=int), 0)

        self._fit_state = _SVMFit(
            weight=weights,
            bias=float(bias),
            support_vectors=sv_tuple[0],
            support=sv_tuple[1],
            n_support=sv_tuple[2],
            classes=classes,
            loss_history=loss_h,
            scaler_mean=scaler[0],
            scaler_scale=scaler[1],
            weights_mc=None,
            biases_mc=None,
            col_medians=col_medians,
        )
        return self

    # Receives: x_data — new feature data to classify.
    # Calculates: the raw decision-function score(s) for every sample
    #   (via `decision_function`), then — for binary classification —
    #   thresholds at 0; for multiclass, picks whichever class's own
    #   binary classifier scored highest.
    # Returns: an array of shape (n_samples,) with the predicted class
    #   label for every sample.
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        scores = self.decision_function(x_data)
        fs = self._fit_state
        if fs is not None and fs.weights_mc is not None:
            # Multiclass OvR: argmax over per-class decision scores.
            # `scores` here has shape (n_samples, n_classes) — one column
            # per class's own binary classifier; whichever column scored
            # highest for a sample is that sample's predicted class.
            return self.classes_[np.argmax(scores, axis=1)]
        # Binary case: a positive score means "positive class" (index 1 of
        # `classes_`), a non-positive score means "negative class" (index 0).
        idx = (scores > 0).astype(int)
        return self.classes_[idx]

    # Receives: x_data — feature data to score, shape (n_samples,
    #   n_features) or (n_features,) for a single sample.
    # Calculates: imputes/scales/kernel-transforms x_data the exact same
    #   way training data was, then computes the raw (unthresholded) SVM
    #   decision score — how far on the positive or negative side of the
    #   hyperplane each sample falls.
    # Returns: for a binary model, an array of shape (n_samples,); for a
    #   multiclass (OvR) model, an array of shape (n_samples, n_classes)
    #   with one score column per class.
    def decision_function(self, x_data: np.ndarray) -> np.ndarray:
        """Signed distance to the decision hyperplane.

        For binary classification returns shape ``(n_samples,)``.
        For OvR multiclass returns shape ``(n_samples, n_classes)``.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        scores : np.ndarray
        """
        x_data = np.asarray(x_data, dtype=np.float64)
        if x_data.ndim == 1:
            x_data = x_data.reshape(1, -1)
        fs = self._fit_state
        if fs is not None and np.isnan(x_data).any():
            # Fill any missing values using the SAME per-column medians
            # computed at fit time (never medians of the new data itself —
            # that would let test-time statistics leak into predictions).
            x_data = x_data.copy()
            fill = fs.col_medians if fs.col_medians is not None else np.zeros(x_data.shape[1])
            nan_pos = np.isnan(x_data)
            x_data[nan_pos] = np.take(fill, np.where(nan_pos)[1])
        if fs is not None and fs.scaler_mean is not None:
            # Apply the exact same standardisation (mean/scale) learned
            # from the training data.
            x_data = (x_data - fs.scaler_mean) / fs.scaler_scale
        if self._cfg.kernel == "rbf":
            x_data = np.exp(-self._cfg.gamma * x_data)
        if fs is not None and fs.weights_mc is not None:
            # Multiclass: one dot product per class, all done in a single
            # matrix multiply — column i of the result is class i's own
            # binary classifier's score for every sample.
            return x_data @ fs.weights_mc.T + fs.biases_mc
        return x_data @ self.weight_ + self.bias_

    # Receives: x_data — test feature data, and y — the true labels for
    #   that same data.
    # Calculates: the model's predictions for x_data, compared against the
    #   true labels y.
    # Returns: a single float — the fraction of samples classified
    #   correctly.
    def score(self, x_data: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy on (x_data, y).

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        accuracy : float
        """
        return float(np.mean(self.predict(x_data) == y))

    # Receives: x_data and y — new data to stream in after the model has
    #   already been fitted once, and chunk_size — how many samples to
    #   process per mini-batch update.
    # Calculates: splits (x_data, y) into consecutive mini-batches (via
    #   `_batch_generator`), and applies one sub-gradient update per batch
    #   (via `_update_step`) — or, in multiclass mode, simply re-runs the
    #   full `fit()` on each batch, since the OvR training loop doesn't
    #   have its own incremental update path.
    # Returns: self, with `self._fit_state` updated after processing every
    #   batch.
    def partial_fit(
        self, x_data: np.ndarray, y: np.ndarray, chunk_size: int = 32
    ) -> "SVM":
        """Apply sub-gradient updates on each mini-batch chunk (data streaming).

        Splits (x_data, y) into chunks using :meth:`_batch_generator` (a generator
        that yields successive mini-batches), then calls :meth:`_update_step` on
        each chunk.  Requires :meth:`fit` to have been called first.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)
            New data to stream through the model.
        y : np.ndarray of shape (n_samples,)
            Corresponding binary class labels.
        chunk_size : int, default=32
            Number of samples per mini-batch.

        Returns
        -------
        self
        """
        if self._fit_state is None:
            raise ValueError("Call fit() before partial_fit().")
        x_data = np.asarray(x_data, dtype=np.float64)
        y = np.asarray(y)
        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)
        if self._fit_state.weights_mc is not None:
            # Multiclass OvR: stream batches through fit() (data streaming).
            for x_b, y_b in self._batch_generator(x_data, y, chunk_size):
                self.fit(x_b, y_b)
            return self
        for x_b, y_b in self._batch_generator(x_data, y, chunk_size):
            self._update_step(x_b, y_b)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Receives: x_data and y — a dataset to split into pieces, and
    #   chunk_size — the target size of each piece.
    # Calculates: nothing — this is a generator, meaning it produces each
    #   chunk lazily, on demand, one at a time, rather than building the
    #   whole list of chunks upfront.
    # Returns (yields): successive (x_chunk, y_chunk) pairs covering the
    #   whole dataset, in order.
    @staticmethod
    def _batch_generator(x_data: np.ndarray, y: np.ndarray, chunk_size: int = 32):
        """Yield successive (x_chunk, y_chunk) mini-batches for data streaming.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        chunk_size : int
            Size of each yielded chunk.

        Yields
        ------
        tuple
            ``(x_chunk, y_chunk)`` mini-batch pair.
        """
        for start in range(0, len(x_data), chunk_size):
            yield x_data[start:start + chunk_size], y[start:start + chunk_size]

    # Receives: x_batch and y_batch — one mini-batch of new streaming data
    #   (raw, unscaled).
    # Calculates: scales x_batch using the ALREADY-fitted scaler, encodes
    #   y_batch to {-1, +1}, then computes and applies exactly one
    #   sub-gradient step to the CURRENT weight/bias (the same update math
    #   as one iteration of `_train_loop`, just applied once instead of
    #   looped).
    # Returns: nothing — replaces `self._fit_state` with a new tuple
    #   carrying the updated weight and bias (everything else copied over
    #   unchanged).
    def _update_step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> None:
        """Apply one sub-gradient step on (x_batch, y_batch).

        Parameters
        ----------
        x_batch : np.ndarray of shape (n_samples, n_features)
        y_batch : np.ndarray of shape (n_samples,)
        """
        fs = self._fit_state
        if fs is None:
            return
        x_sc = (x_batch - fs.scaler_mean) / fs.scaler_scale
        y_bin = self._encode(y_batch, fs.classes)
        weights = fs.weight.copy()
        bias = fs.bias
        # The functional margin for each sample: positive means correctly
        # classified with room to spare, <1 means inside the margin or
        # misclassified (a "violator").
        margin = y_bin * (x_sc @ weights + bias)
        violators = margin < 1.0
        # Regularisation gradient applies to every sample regardless of
        # margin; the hinge-loss gradient only comes from violators.
        grad_w = weights / self._cfg.reg_param
        grad_b = 0.0
        if violators.any():
            grad_w -= np.mean(y_bin[violators, None] * x_sc[violators], axis=0)
            grad_b = -float(np.mean(y_bin[violators]))
        weights -= self._cfg.learning_rate * grad_w
        bias -= self._cfg.learning_rate * grad_b
        self._fit_state = _SVMFit(
            weight=weights,
            bias=float(bias),
            support_vectors=fs.support_vectors,
            support=fs.support,
            n_support=fs.n_support,
            classes=fs.classes,
            loss_history=fs.loss_history,
            scaler_mean=fs.scaler_mean,
            scaler_scale=fs.scaler_scale,
            weights_mc=fs.weights_mc,
            biases_mc=fs.biases_mc,
            col_medians=fs.col_medians,
        )

    # Receives: x_feat — the already scaled/kernel-transformed feature
    #   matrix, y — the ORIGINAL (not yet binarised) class labels, and
    #   classes — the sorted array of every distinct class.
    # Calculates: trains all K one-vs-rest binary classifiers TOGETHER in
    #   a single epoch loop, by stacking every class's own {-1,+1} label
    #   vector into one (K, n) matrix and computing every classifier's
    #   margins/gradients with (K, n)-shaped matrix operations instead of
    #   looping over classes and calling `_train_loop` K separate times.
    # Returns: a 3-tuple (weights_mc, biases_mc, loss_history) — weights_mc
    #   has shape (n_classes, n_features), one row per class's own weight
    #   vector; biases_mc has shape (n_classes,); loss_history is the
    #   averaged loss (across all K classifiers) at each epoch.
    def _train_multiclass(
        self, x_feat: np.ndarray, y: np.ndarray, classes: np.ndarray
    ) -> tuple:
        """Vectorised OvR training — all K classifiers in a single epoch loop.

        Replaces K sequential ``_train_loop`` calls with one loop that processes
        all K one-vs-rest label vectors simultaneously using (K, n) matrix ops,
        reducing Python/NumPy dispatch overhead by a factor of K.

        Parameters
        ----------
        x_feat : np.ndarray of shape (n_samples, n_features)
            Already scaled / kernel-transformed feature matrix.
        y : np.ndarray of shape (n_samples,)
            Original class labels (multiple classes).
        classes : np.ndarray
            Sorted unique class values.

        Returns
        -------
        tuple
            ``(weights_mc, biases_mc, loss_history)`` where weights_mc has
            shape ``(n_classes, n_features)`` and biases_mc shape ``(n_classes,)``.
        """
        n_samples = len(y)
        # (K, n): each row is the OvR binary label vector for one class.
        # For class c, every sample belonging to c gets label +1 and every
        # OTHER sample gets -1 — this is exactly what "one class versus
        # the rest" means, stacked into one matrix so all K classes can be
        # trained with the same vectorised operations at once.
        y_bin_mc = np.vstack([np.where(y == c, 1.0, -1.0) for c in classes])
        weights = np.zeros((len(classes), x_feat.shape[1]))
        biases = np.zeros(len(classes))
        loss_history: List[float] = []

        for _epoch in range(self._cfg.n_iterations):
            # (K, n) = (K, d) @ (d, n): BLAS level-3 gemm — one call for all K.
            # Every class's own margin for every sample, computed via a
            # single matrix multiply instead of K separate ones.
            margins = y_bin_mc * (weights @ x_feat.T + biases[:, None])
            loss = float(np.maximum(0.0, 1.0 - margins).sum()) / (len(classes) * n_samples)
            loss_history.append(loss)
            if loss < self._cfg.tol:
                break
            # Vectorised sub-gradient across all K classifiers.
            viol = (margins < 1.0).astype(np.float64)
            grad_w = weights / self._cfg.reg_param - (y_bin_mc * viol) @ x_feat / n_samples
            lr = (
                self._cfg.learning_rate / np.sqrt(_epoch + 1)
                if self._cfg.lr_decay
                else self._cfg.learning_rate
            )
            biases -= lr * (-(y_bin_mc * viol).sum(axis=1) / n_samples)
            weights -= lr * grad_w

        return weights, biases, loss_history

    # Receives: x_feat — the scaled/kernel-transformed feature matrix, and
    #   y_bin — the binary labels already encoded as {-1, +1}.
    # Calculates: runs full-batch sub-gradient descent for up to
    #   `n_iterations` epochs: each epoch computes every sample's margin,
    #   the average hinge loss, and (for samples violating the margin) the
    #   gradient of the regularised hinge-loss objective, then takes one
    #   step against that gradient. Stops early if the loss drops below
    #   `tol`.
    # Returns: a 3-tuple (weights, bias, loss_history) — the final learned
    #   weight vector and bias, plus the per-epoch loss values recorded
    #   along the way.
    def _train_loop(
        self, x_feat: np.ndarray, y_bin: np.ndarray
    ) -> tuple:
        """Run the sub-gradient descent loop; return (weights, bias, loss_history).

        Parameters
        ----------
        x_feat : np.ndarray of shape (n_samples, n_features)
            Feature matrix (already scaled / kernel-transformed).
        y_bin : np.ndarray of shape (n_samples,)
            Binary labels in {-1, +1}.

        Returns
        -------
        tuple
            ``(weights, bias, loss_history)``
        """
        n_samples = len(y_bin)
        weights = np.zeros(x_feat.shape[1])
        bias = 0.0
        loss_history: List[float] = []

        for epoch in range(self._cfg.n_iterations):
            # Functional margin for every sample at once: positive and
            # large means "confidently correct", less than 1 means inside
            # the margin (or outright wrong).
            margin = y_bin * (x_feat @ weights + bias)
            # Use sum+divide instead of np.mean to reduce Python dispatch overhead.
            hinge = np.maximum(0.0, 1.0 - margin)
            loss = float(hinge.sum()) / n_samples
            loss_history.append(loss)
            if loss < self._cfg.tol:
                if self._cfg.verbose:
                    print(f"SVM converged at epoch {epoch}")
                break

            violators = margin < 1.0
            grad_w = weights / self._cfg.reg_param
            grad_b = 0.0
            if violators.any():
                # sub-gradient: sum over violators / n_total (correct normalisation).
                # Only the samples that actually violate the margin
                # contribute to the hinge-loss part of the gradient — a
                # sample already safely outside the margin has zero hinge
                # gradient.
                grad_w -= y_bin[violators] @ x_feat[violators] / n_samples
                grad_b -= float(y_bin[violators].sum()) / n_samples

            lr = (
                self._cfg.learning_rate / np.sqrt(epoch + 1)
                if self._cfg.lr_decay
                else self._cfg.learning_rate
            )
            weights -= lr * grad_w
            bias -= lr * grad_b

        return weights, bias, loss_history

    # Receives: y — the raw class labels (any two distinct values), and
    #   classes — the sorted array of those two distinct values.
    # Calculates: maps whichever value comes FIRST in `classes` (the
    #   smaller one, since classes is sorted) to -1, and the other value
    #   to +1.
    # Returns: an array of {-1.0, +1.0} labels, same length as y.
    @staticmethod
    def _encode(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Map binary labels to {-1, +1} using the order in ``classes``."""
        return np.where(y == classes[0], -1.0, 1.0)

    # Receives: x_orig — the (scaled but not kernel-transformed) original
    #   feature values, y_bin — the {-1,+1} training labels, x_feat — the
    #   feature-mapped values actually used for training, and the learned
    #   weights/bias.
    # Calculates: for every training sample, its decision score under the
    #   fitted model, then flags any sample whose margin is <= 1 (i.e. on
    #   or inside the margin, or misclassified) as a support vector.
    # Returns: a 3-tuple (sv_X, sv_indices, n_sv) — the support vectors'
    #   own feature rows, their row indices in the training set, and how
    #   many there are.
    @staticmethod
    def _find_support_vectors(
        x_orig: np.ndarray,
        y_bin: np.ndarray,
        x_feat: np.ndarray,
        weights: np.ndarray,
        bias: float,
    ) -> tuple:
        """Identify support vectors; return (sv_X, sv_indices, n_sv)."""
        if weights.size == 0 or len(x_feat) == 0:
            return x_orig[:0], np.array([], dtype=int), 0
        scores = x_feat @ weights + bias
        # A small epsilon (1e-5) tolerance accounts for floating-point
        # rounding right at the margin boundary, so a sample that's
        # mathematically exactly on the margin isn't excluded due to a
        # tiny numerical overshoot.
        sv_mask = (y_bin * scores) <= 1.0 + 1e-5
        return x_orig[sv_mask], np.where(sv_mask)[0], int(sv_mask.sum())
