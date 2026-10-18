"""Binary logistic regression — batch and mini-batch SGD variants.

Both classifiers implement the standard logistic regression model
    h_w(x) = σ(wᵀx + b),  σ(z) = 1 / (1 + e⁻ᶻ)
optimised by minimising the cross-entropy loss
    J(w) = -(1/m) Σ [ y·log(h) + (1-y)·log(1-h) ] + (α/2)·||w||²
using the gradient
    ∇_w J = (1/m) Xᵀ(h - y) + α·w
    ∇_b J = (1/m) Σ (h - y)

Reference: 03-logistic-regression.ipynb (course notebook), Sections 3–5.
"""

import numpy as np
from typing import Optional


# Numerical-stability constants — module-level so both classes share them.
_SIGMOID_CLIP: float = 500.0   # clip z to keep np.exp(-z) finite
_LOSS_EPS: float = 1e-15        # additive epsilon to keep log(p) finite
_AUTO_BALANCE_RATIO: float = 3.0   # imbalance ratio above which 'auto' kicks in



# Receives: z — an array of "pre-activation" values (i.e. wᵀx + b, before
#   the sigmoid squashes it into a probability), any shape.
# Calculates: the logistic sigmoid of every entry, after first clamping z to
#   a safe range so the exponential inside it can never overflow.
# Returns: an array the same shape as z, with every value squashed into the
#   (0, 1) range.
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid σ(z) = 1 / (1 + e⁻ᶻ).

    Clipping z to [-500, 500] prevents `np.exp` overflow for extreme inputs;
    σ saturates at 0/1 well before that bound so the truncation is invisible.

    Parameters
    ----------
    z : np.ndarray
        Pre-activation values of any shape.

    Returns
    -------
    np.ndarray
        Element-wise sigmoid of `z`, same shape as input.
    """
    # np.clip(z, -500, 500) forces every value of z into that range before
    # anything else happens — if z were allowed to be, say, -10000, then
    # np.exp(-z) would try to compute e^10000, which overflows to infinity
    # and would break the division below. Since sigmoid already flattens
    # out to essentially 0 or 1 well before z reaches ±500, clamping it
    # there changes nothing about the actual math, only prevents the
    # overflow.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -_SIGMOID_CLIP, _SIGMOID_CLIP)))


# Receives: y_true — the actual binary labels (0 or 1) for every sample, and
#   y_prob — the model's predicted probability that each sample is class 1.
# Calculates: the average binary cross-entropy loss across all samples,
#   after clipping predicted probabilities away from exactly 0 or 1.
# Returns: a single float — the mean cross-entropy loss.
def _log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean binary cross-entropy loss with numerical-stability clipping.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth labels in {0, 1}.
    y_prob : np.ndarray of shape (n_samples,)
        Predicted P(y=1 | x).

    Returns
    -------
    float
        Mean cross-entropy. Probabilities are clipped to [eps, 1-eps] to
        avoid log(0) = -inf.
    """
    # If a predicted probability were exactly 0 or 1, np.log(0) would be
    # -infinity, which would break the loss computation for that sample (and
    # potentially the whole training run). Clipping every probability to
    # just inside (0, 1) — using a tiny epsilon — avoids that without
    # meaningfully changing the loss value for any sample that wasn't
    # already at an extreme.
    p = np.clip(y_prob, _LOSS_EPS, 1.0 - _LOSS_EPS)
    # This is the standard binary cross-entropy formula: for each sample,
    # if the true label is 1, only the `log(p)` term matters (rewarding
    # confident correct predictions and punishing confident wrong ones);
    # if the true label is 0, only the `log(1-p)` term matters. Averaging
    # (with a leading minus sign, since loss should be positive when the
    # model is wrong) over every sample gives the overall loss.
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


# Receives: X — the raw feature data, and y — the raw target labels, in
#   whatever array-like form the caller passed them in.
# Calculates: converts both to proper numpy arrays, then checks that X is
#   2D, that neither X nor y is empty, that their row counts match, and
#   that y only contains the labels 0 and 1.
# Returns: a tuple (X_arr, y_arr) — the cleaned-up, validated arrays; raises
#   ValueError immediately if any check fails.
def _validate_inputs(X: np.ndarray, y: np.ndarray) -> tuple:
    """Validate and normalise (X, y) for binary classification.

    Ensures X is 2D float, y is 1D, lengths match, and y holds only
    binary labels (0/1). Returns the cleaned arrays.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target labels.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Validated `X` (float, 2D) and `y` (1D).

    Raises
    ------
    ValueError
        If shapes mismatch or y contains classes other than {0, 1}.
    """
    # Normalise both inputs into plain numpy arrays: X as 2D floats, y
    # flattened to a simple 1D list of labels (`.ravel()` flattens any
    # accidental extra dimensions away).
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).ravel()

    # `.ndim` counts the array's dimensions — a feature matrix must be 2D
    # (rows = samples, columns = features).
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features); got shape {X_arr.shape}")
    # Empty-data guard — must come before the binary-label check (np.unique on
    # an empty array vacuously satisfies "all in {0, 1}", letting fit produce
    # NaN weights via 0/0 in the gradient).
    if X_arr.shape[0] == 0:
        raise ValueError(
            "Empty training data: X has 0 samples. "
            "Provide at least one sample with both classes ideally represented."
        )
    if y_arr.shape[0] == 0:
        raise ValueError(
            "Empty training data: y has 0 labels. "
            "Provide at least one labelled sample to fit a model."
        )
    # X and y must describe the same set of samples, one row of X per entry
    # of y.
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X and y row counts disagree: {X_arr.shape[0]} vs {y_arr.shape[0]}"
        )

    # np.unique(y_arr) lists every distinct label value that actually shows
    # up in y. np.isin(classes, [0, 1]) checks, for each of those distinct
    # values, whether it's either 0 or 1; np.all(...) then confirms EVERY
    # distinct value passed that check — i.e. there's no label besides 0
    # and 1 anywhere in y.
    classes = np.unique(y_arr)
    if not np.all(np.isin(classes, [0, 1])):
        raise ValueError(
            f"y must contain only binary labels {{0, 1}}; got classes {classes.tolist()}"
        )

    return X_arr, y_arr.astype(float)


# Receives: y — the binary training labels, and class_weight — one of
#   None / 'auto' / 'balanced' / a {label: weight} dict, describing how
#   much extra importance to give each class during training.
# Calculates: works out, for every sample, a weight to multiply its
#   gradient contribution by — larger for the minority class when
#   rebalancing is requested, so a lopsided dataset doesn't get ignored by
#   the model.
# Returns: an array of shape (n_samples,), one weight per sample, all 1.0
#   when no rebalancing applies.
def _compute_sample_weights(y: np.ndarray, class_weight) -> np.ndarray:
    """Per-sample weights for handling class imbalance.

    Mirrors scikit-learn's ``class_weight`` semantics, plus an extra
    ``'auto'`` mode that only re-weights when the data is *actually*
    imbalanced. This lets the default behaviour stay sklearn-compatible
    on roughly balanced data while still rescuing the minority class on
    heavily skewed inputs.

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)
        Binary labels in {0, 1}.
    class_weight : None, 'auto', 'balanced', or dict
        - ``'auto'``         — inspect the class ratio. If the larger class
          is more than ``_AUTO_BALANCE_RATIO`` (= 3) times the smaller, fall
          back to ``'balanced'``; otherwise return all ones. This is the
          library's default.
        - ``None``           — every sample weighted 1.0 (no rebalancing).
        - ``'balanced'``     — always apply inverse class frequency
          ``n / (2·count_c)`` (the sklearn formula), regardless of the
          actual class ratio.
        - ``{0: w0, 1: w1}`` — explicit per-class weights.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Sample weight for every row of `y`.

    Raises
    ------
    ValueError
        If `class_weight` is anything other than the four accepted forms.
    """
    if class_weight is None:
        # No rebalancing requested — every sample counts equally.
        return np.ones_like(y, dtype=float)

    if class_weight == "auto":
        # Count how many samples belong to each class.
        n0 = float(np.sum(y == 0))
        n1 = float(np.sum(y == 1))
        # If a class is missing entirely or the ratio is mild, no rebalancing.
        if n0 == 0 or n1 == 0:
            return np.ones_like(y, dtype=float)
        # How many times bigger is the larger class than the smaller one?
        ratio = max(n0, n1) / min(n0, n1)
        if ratio < _AUTO_BALANCE_RATIO:
            # Close enough to balanced — don't bother rebalancing.
            return np.ones_like(y, dtype=float)
        # Severe imbalance — fall through to the 'balanced' branch.
        class_weight = "balanced"

    if class_weight == "balanced":
        # sklearn's standard "balanced" formula: each class's weight is
        # inversely proportional to how common it is, so a rare class's
        # samples individually count for more, cancelling out its smaller
        # numbers overall.
        n_samples = y.shape[0]
        n0 = float(np.sum(y == 0))
        n1 = float(np.sum(y == 1))
        w0 = n_samples / (2.0 * n0) if n0 > 0 else 1.0
        w1 = n_samples / (2.0 * n1) if n1 > 0 else 1.0
        # np.where(condition, value_if_true, value_if_false), applied
        # element-by-element: give every sample labelled 1 the weight w1,
        # and every sample labelled 0 the weight w0.
        return np.where(y == 1, w1, w0).astype(float)

    if isinstance(class_weight, dict):
        # Caller supplied explicit weights per label — look each sample's
        # own label up in that dict (defaulting to 1.0 if a label was
        # somehow left out).
        return np.array(
            [float(class_weight.get(int(label), 1.0)) for label in y], dtype=float
        )

    raise ValueError(
        f"class_weight must be None, 'auto', 'balanced', or a dict; got {class_weight!r}"
    )


class LogisticRegression:  # pylint: disable=too-many-instance-attributes
    """Binary logistic regression trained via batch gradient descent.

    Each iteration uses the **entire** training set to compute the gradient
    (full-batch GD). Suitable for small/medium datasets where one pass over
    all samples per update is affordable.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size η for each gradient update.
    n_iterations : int, default=1000
        Number of full-batch gradient steps.
    alpha : float, default=0.0
        L2 regularization strength on the weights (bias is not penalised).
    class_weight : None, 'auto', 'balanced', or dict, default='auto'
        Per-class weighting for handling **class imbalance**.
        - ``'auto'`` (default) → inspect class ratio at fit time. If the
          majority class is > 3× the minority, apply ``'balanced'`` weights;
          otherwise leave the gradient unweighted. This keeps roughly
          balanced datasets sklearn-compatible while still handling
          severely skewed inputs.
        - ``None``           → unweighted (every sample weight = 1).
        - ``'balanced'``     → always apply inverse class frequency
          (sklearn semantics) regardless of actual ratio.
        - ``{0: w0, 1: w1}`` → explicit per-class weights.
    random_state : int or None, default=None
        Seed for NumPy's RNG — set for reproducible runs.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_features,)
        Learned feature weights. Set after fit().
    bias_ : float
        Learned intercept term. Set after fit().
    coef_ : np.ndarray of shape (n_features,)
        Alias of `weights_` (scikit-learn compatible).
    intercept_ : float
        Alias of `bias_` (scikit-learn compatible).
    classes_ : np.ndarray of shape (2,)
        Sorted unique class labels seen during fit (always [0, 1] here).
    n_iter_ : int
        Number of iterations actually run.
    loss_history_ : list of float
        Mean cross-entropy (with L2 penalty if α>0) per iteration. Use this
        to verify convergence — values should monotonically decrease.

    Notes
    -----
    Feature scaling (e.g. ``StandardScaler``) is recommended for faster
    convergence, especially when feature scales differ by orders of magnitude.
    Use **training-set statistics only** when scaling test data.
    """

    # Receives: learning_rate (step size for each gradient update),
    #   n_iterations (how many full-batch gradient steps to run), alpha (L2
    #   regularization strength), class_weight (how to handle class
    #   imbalance — see `_compute_sample_weights`), random_state (seed for
    #   reproducibility).
    # Calculates: nothing yet — just remembers every hyperparameter and
    #   sets every fitted attribute to a placeholder until fit() is called.
    # Returns: nothing — __init__ only prepares the object.
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        alpha: float = 0.0,
        class_weight="auto",
        random_state: Optional[int] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.class_weight = class_weight
        self.random_state = random_state

        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_iter_: Optional[int] = None
        self.loss_history_: list = []

    # Receives: X — the training feature matrix, and y — the binary target
    #   labels (0 or 1) for every sample.
    # Calculates: repeatedly computes the model's current predicted
    #   probabilities for the WHOLE training set at once, works out the
    #   (possibly class-weighted) gradient of the cross-entropy loss, and
    #   nudges `weights_`/`bias_` a small step against that gradient — for
    #   `n_iterations` passes.
    # Returns: self (the fitted model), and stores `weights_`, `bias_`,
    #   `coef_`, `intercept_`, `classes_`, `n_iter_`, and `loss_history_` as
    #   attributes on `self` along the way.
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit the model by minimising regularised cross-entropy with batch GD.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix (apply feature scaling before fitting).
        y : np.ndarray of shape (n_samples,)
            Binary target labels in {0, 1}.

        Returns
        -------
        self : LogisticRegression
            Fitted estimator (enables `model.fit(X, y).predict(X_test)` chaining).
        """
        X_arr, y_arr = _validate_inputs(X, y)
        # `_` throws away the sample count (not needed here); `n_features`
        # is how many weights we'll need to learn.
        _, n_features = X_arr.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Per-sample weights (handles class imbalance — see _compute_sample_weights).
        # All ones if class_weight is None → identical behaviour to the unweighted case.
        sample_weights = _compute_sample_weights(y_arr, self.class_weight)

        # Reset state on every fit() — repeated fits must not accumulate.
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.classes_ = np.array([0, 1])
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            # Forward pass: predicted probabilities for the full batch.
            # `X_arr.dot(self.weights_) + self.bias_` computes wᵀx + b for
            # every sample at once; `_sigmoid(...)` turns each of those raw
            # scores into a probability between 0 and 1.
            probs = _sigmoid(X_arr.dot(self.weights_) + self.bias_)
            # Multiplying the residual by sample_weights re-weights each sample's
            # contribution to the gradient → minority-class samples pull harder
            # when class_weight='balanced'.
            # `probs - y_arr` is how far off each prediction is from the
            # true label (0 or 1) — the raw error.
            weighted_error = (probs - y_arr) * sample_weights

            # Gradient (vectorised). Bias gradient excludes regularisation.
            # `X_arr.T.dot(weighted_error)` sums each feature's contribution
            # to the error across every sample at once; dividing by the
            # sample count averages it, and `self.alpha * self.weights_`
            # adds the L2 penalty's own gradient (proportional to the
            # current weight).
            grad_w = X_arr.T.dot(weighted_error) / X_arr.shape[0] + self.alpha * self.weights_
            # The bias gradient is just the average (weighted) error — no
            # per-feature multiplication needed, and no regularisation
            # (the bias is never penalised).
            grad_b = float(np.mean(weighted_error))

            # Step both parameters a small distance (scaled by
            # `learning_rate`) in the direction that REDUCES the loss —
            # i.e. opposite to the gradient, which points toward increasing
            # loss.
            self.weights_ -= self.learning_rate * grad_w
            self.bias_    -= self.learning_rate * grad_b

            # Track regularised loss for convergence monitoring.
            loss = _log_loss(y_arr, probs)
            if self.alpha > 0.0:
                # Add the L2 penalty term itself (not just its gradient) to
                # the reported loss, so the logged value reflects the FULL
                # cost function actually being minimised.
                loss += 0.5 * self.alpha * float(np.sum(self.weights_ ** 2))
            self.loss_history_.append(loss)

        # Sklearn-compatible aliases — set after the loop so they always
        # reflect the final fitted state.
        self.coef_ = self.weights_.copy()
        self.intercept_ = self.bias_
        self.n_iter_ = self.n_iterations
        return self

    # Receives: X — feature data to predict on, shape (n_samples,
    #   n_features).
    # Calculates: the model's predicted probability of each class, for
    #   every sample, using the weights/bias learned in fit().
    # Returns: an array of shape (n_samples, 2) — column 0 is P(y=0|x),
    #   column 1 is P(y=1|x); each row sums to 1.
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict per-class probabilities P(y=0 | x) and P(y=1 | x).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Column 0 holds P(y=0|x); column 1 holds P(y=1|x). Rows sum to 1.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X_arr.shape}")
        if self.weights_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        # P(y=1|x) is the model's raw sigmoid output.
        p1 = _sigmoid(X_arr.dot(self.weights_) + self.bias_)
        # np.column_stack glues separate 1D arrays together as columns of a
        # 2D array. Since P(y=0|x) and P(y=1|x) must add up to 1 for a
        # binary classifier, P(y=0|x) is simply `1 - p1` — no separate
        # computation needed.
        return np.column_stack([1.0 - p1, p1])

    # Receives: X — feature data to predict on, and threshold — the cutoff
    #   probability above which a sample is called class 1.
    # Calculates: the predicted probability of class 1 for every sample
    #   (via predict_proba), then compares each one against `threshold`.
    # Returns: an array of shape (n_samples,) with hard 0/1 class
    #   predictions.
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels by thresholding P(y=1 | x).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        threshold : float, default=0.5
            Decision threshold on P(y=1 | x). Values in [0, 1].

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels in {0, 1}.
        """
        # `[:, 1]` selects just the "P(y=1|x)" column from predict_proba's
        # output; comparing it to `threshold` gives a True/False array,
        # and `.astype(int)` turns True/False into 1/0.
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    # Receives: X — test feature data, and y — the true labels for that
    #   same data.
    # Calculates: the model's hard predictions for X, then compares them
    #   against the true labels y.
    # Returns: a single float — the fraction of samples the model got
    #   exactly right.
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean classification accuracy on the given test set.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
            True binary labels.

        Returns
        -------
        float
            Fraction of correctly classified samples in [0, 1].
        """
        # `self.predict(X) == np.asarray(y).ravel()` compares predictions to
        # truth element-by-element; np.mean over that boolean array counts
        # True as 1 and False as 0, giving exactly the accuracy fraction.
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))


class SGDClassifier:  # pylint: disable=too-many-instance-attributes
    """Binary logistic classifier trained via mini-batch Stochastic GD.

    At every epoch the data is shuffled and split into mini-batches of size
    `batch_size`. Each mini-batch produces one gradient update — many updates
    per epoch with noisier but faster progress than full-batch GD.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size η for each mini-batch update.
    n_iterations : int, default=1000
        Number of epochs (full passes over the training data).
    batch_size : int, default=32
        Samples per mini-batch. Smaller → noisier gradients, more updates.
    alpha : float, default=0.0
        L2 regularization strength on the weights (bias not penalised).
    class_weight : None, 'auto', 'balanced', or dict, default='auto'
        Per-class weighting for handling **class imbalance**. ``'auto'``
        (default) only re-weights when the ratio between majority and
        minority class exceeds 3:1; otherwise the gradient is left
        unweighted (matches sklearn behaviour on roughly balanced data).
        Pass ``'balanced'`` to force inverse-class-frequency weighting,
        ``None`` to disable, or a ``{0: w0, 1: w1}`` dict to set
        per-class weights explicitly.
    random_state : int or None, default=None
        Seed for the shuffle RNG — set for reproducible runs.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_features,)
        Learned feature weights. Set after fit().
    bias_ : float
        Learned intercept term. Set after fit().
    coef_ : np.ndarray of shape (n_features,)
        Alias of `weights_` (scikit-learn compatible).
    intercept_ : float
        Alias of `bias_` (scikit-learn compatible).
    classes_ : np.ndarray of shape (2,)
        Sorted unique class labels seen during fit.
    n_iter_ : int
        Number of epochs actually run.
    loss_history_ : list of float
        Mean cross-entropy per epoch (averaged over mini-batches). Useful
        for plotting convergence; expect noisy decrease.

    Notes
    -----
    Feature scaling is **strongly recommended** before fitting — gradient
    descent is very sensitive to feature scale. Standardise using training
    statistics only.
    """

    # Receives: learning_rate, n_iterations (here: epochs), batch_size (rows
    #   per mini-batch), alpha (L2 strength), class_weight (imbalance
    #   handling), random_state (seed for the shuffle RNG).
    # Calculates: nothing yet — remembers every hyperparameter and sets
    #   every fitted attribute to a placeholder until fit() runs.
    # Returns: nothing — __init__ only prepares the object.
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        batch_size: int = 32,
        alpha: float = 0.0,
        class_weight="auto",
        random_state: Optional[int] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.alpha = alpha
        self.class_weight = class_weight
        self.random_state = random_state

        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_iter_: Optional[int] = None
        self.loss_history_: list = []

    # Receives: X — the (ideally pre-scaled) training feature matrix, and
    #   y — the binary target labels for every sample.
    # Calculates: for each epoch, shuffles the data, splits it into
    #   mini-batches, and updates `weights_`/`bias_` once per mini-batch
    #   using that batch's (possibly class-weighted) cross-entropy
    #   gradient — repeated for `n_iterations` epochs.
    # Returns: self (the fitted model), storing `weights_`, `bias_`,
    #   `coef_`, `intercept_`, `classes_`, `n_iter_`, and `loss_history_` as
    #   attributes along the way.
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGDClassifier":
        """Fit the model with mini-batch stochastic gradient descent.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Pre-scaled training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Binary target labels in {0, 1}.

        Returns
        -------
        self : SGDClassifier
            Fitted estimator.
        """
        X_arr, y_arr = _validate_inputs(X, y)
        n_samples, n_features = X_arr.shape

        # A dedicated random generator seeded with `random_state`, so the
        # shuffle order is reproducible given the same seed.
        rng = np.random.default_rng(self.random_state)

        # Per-sample weights (handles class imbalance — see _compute_sample_weights).
        # All ones when class_weight is None → identical to the unweighted case.
        sample_weights_full = _compute_sample_weights(y_arr, self.class_weight)

        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.classes_ = np.array([0, 1])
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            # Shuffle each epoch to break sample ordering bias.
            # rng.permutation(n_samples) gives a random reordering of the
            # row indices; indexing X, y and the sample weights with the
            # SAME `idx` keeps every sample's feature row, label, and
            # weight correctly matched up after shuffling.
            idx = rng.permutation(n_samples)
            X_shuffled = X_arr[idx]
            y_shuffled = y_arr[idx]
            sw_shuffled = sample_weights_full[idx]   # keep weights aligned with samples

            batch_losses: list = []
            for start in range(0, n_samples, self.batch_size):
                # Slice out one mini-batch's worth of rows (Python slicing
                # safely stops at the end of the array, so the last batch
                # of an epoch can be smaller than `batch_size`).
                X_batch = X_shuffled[start : start + self.batch_size]
                y_batch = y_shuffled[start : start + self.batch_size]
                sw_batch = sw_shuffled[start : start + self.batch_size]
                m = X_batch.shape[0]

                # Forward pass on the mini-batch.
                probs = _sigmoid(X_batch.dot(self.weights_) + self.bias_)
                # Weight the residual per sample so minority-class samples
                # contribute proportionally more to the gradient.
                weighted_error = (probs - y_batch) * sw_batch

                # Mini-batch gradient (vectorised) — same formula as the
                # full-batch version in LogisticRegression.fit, just
                # computed over this one mini-batch of `m` samples instead
                # of the whole dataset.
                grad_w = X_batch.T.dot(weighted_error) / m + self.alpha * self.weights_
                grad_b = float(np.mean(weighted_error))

                self.weights_ -= self.learning_rate * grad_w
                self.bias_    -= self.learning_rate * grad_b

                # Record this mini-batch's own loss for the epoch-level log
                # below.
                batch_losses.append(_log_loss(y_batch, probs))

            # Average every mini-batch's loss into one number per epoch, so
            # `loss_history_` ends up with one entry per epoch (not one per
            # mini-batch).
            self.loss_history_.append(float(np.mean(batch_losses)))

        self.coef_ = self.weights_.copy()
        self.intercept_ = self.bias_
        self.n_iter_ = self.n_iterations
        return self

    # Receives: X — feature data to predict on, shape (n_samples,
    #   n_features).
    # Calculates: the model's predicted probability of each class for every
    #   sample — identical logic to `LogisticRegression.predict_proba`.
    # Returns: an array of shape (n_samples, 2) with P(y=0|x) and P(y=1|x)
    #   in columns 0 and 1.
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict per-class probabilities — see `LogisticRegression.predict_proba`.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X_arr.shape}")
        if self.weights_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        p1 = _sigmoid(X_arr.dot(self.weights_) + self.bias_)
        return np.column_stack([1.0 - p1, p1])

    # Receives: X — feature data to predict on, and threshold — the cutoff
    #   probability above which a sample is called class 1.
    # Calculates: P(y=1|x) via predict_proba, then compares it to
    #   `threshold`.
    # Returns: an array of shape (n_samples,) with hard 0/1 class labels.
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels by thresholding P(y=1 | x).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        threshold : float, default=0.5

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels in {0, 1}.
        """
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    # Receives: X — test feature data, and y — the true labels for that
    #   same data.
    # Calculates: the model's hard predictions for X, compared against the
    #   true labels y.
    # Returns: a single float — the fraction of samples predicted correctly.
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean classification accuracy on the given test set.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        float
            Accuracy in [0, 1].
        """
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))
