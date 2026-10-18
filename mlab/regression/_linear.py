"""Linear regression: closed-form Normal Equation and mini-batch SGD."""

import numpy as np
from typing import Optional


class LinearRegressor:
    """Linear regression via the Normal Equation.

    Handles both simple (1 feature) and multiple (n features) regression
    using a single unified implementation. Supports optional L2 (Ridge)
    regularization for numerical stability and overfitting prevention.

    Implementation style adapted from:
    https://www.geeksforgeeks.org/machine-learning/linear-regression-python-implementation/

    Parameters
    ----------
    alpha : float, default=0.0
        L2 regularization strength. 0 = plain OLS; larger values shrink coefficients.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_features + 1,)
        Full weight vector [w0, w1, ..., wn]. Set after fit().
    intercept_ : float
        Bias term w0. Set after fit().
    coefficients_ : np.ndarray of shape (n_features,)
        Feature weights [w1, ..., wn]. Set after fit().
    coef_ : np.ndarray of shape (n_features,)
        Alias for coefficients_ (scikit-learn compatible). Set after fit().
    r2score_ : float
        R² score evaluated on the training data. Set after fit().
    """

    # Receives: alpha — the L2 (Ridge) regularization strength to use for
    #   every future call to fit().
    # Calculates: nothing yet — just remembers alpha and sets every fitted
    #   attribute to None as a placeholder until fit() actually runs.
    # Returns: nothing — __init__ only prepares the object.
    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha
        self.weights_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.r2score_: Optional[float] = None

    # Receives: X — the training feature matrix (one row per sample), and
    #   y — the true target value for each sample.
    # Calculates: the exact optimal weight vector via the closed-form Normal
    #   Equation (no iteration needed), then the R² score of the fitted
    #   model on this same training data.
    # Returns: self — after storing `weights_`, `intercept_`,
    #   `coefficients_`, `coef_`, and `r2score_` as attributes on `self`.
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressor":
        """Train the model using the Normal Equation.

        Works for 1 feature (simple LR) and multiple features (multiple LR)
        — the math is identical in both cases.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples,) or (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        LinearRegressor
            The fitted model (allows ``model.fit(X, y).predict(X)``).
        """
        # `.ndim` tells us how many axes the array has. If X came in as a
        # flat (n,) list of single numbers (one feature per sample), reshape
        # it into an (n, 1) column so every line below can always assume a
        # proper 2D feature matrix, whether there's 1 feature or many.
        if X.ndim == 1:
            X = X.reshape(-1, 1)          # simple LR: turn (n,) into (n, 1)

        # `n` is the number of training samples (rows of X).
        n = X.shape[0]
        # `np.c_[...]` stacks arrays side by side as columns. Here it glues a
        # column of all-ones onto the left of X, so the model's bias term
        # (the intercept) can be learned as just another weight, multiplied
        # by a feature that's always 1 — this is the standard trick that
        # lets `X_b.dot(weights)` compute `w0 + w1*x1 + w2*x2 + ...` in one
        # matrix multiply instead of handling the bias separately.
        X_b = np.c_[np.ones((n, 1)), X]  # prepend bias column of ones

        # Ridge Normal Equation: w = (X_bᵀ X_b + αI)⁻¹ X_bᵀ y
        # α=0 reduces to plain OLS; bias term is excluded from regularization.
        # `n_params` is how many weights we're solving for (1 bias + every
        # feature's weight).
        n_params: int = X_b.shape[1]
        # `np.eye(n_params)` builds an identity matrix (1s on the diagonal,
        # 0s elsewhere); multiplying by `self.alpha` scales that whole
        # diagonal by the regularization strength. This is the "αI" term in
        # Ridge regression — it discourages large weights by adding a
        # penalty proportional to each weight's own square.
        reg_matrix: np.ndarray = self.alpha * np.eye(n_params)
        reg_matrix[0, 0] = 0.0  # do not penalise the bias term

        # np.linalg.solve is numerically more stable than explicit matrix inversion.
        # This directly solves the linear system "(X_bᵀX_b + αI) · w = X_bᵀy"
        # for `w`, which is mathematically the same as computing
        # `w = (X_bᵀX_b + αI)⁻¹ · X_bᵀy` but avoids the extra numerical error
        # that comes from actually forming a matrix inverse.
        self.weights_ = np.linalg.solve(
            X_b.T.dot(X_b) + reg_matrix, X_b.T.dot(y)
        )

        # Split the combined weight vector back into its named pieces: the
        # very first entry is the bias/intercept, and everything after it is
        # the per-feature coefficients.
        self.intercept_ = float(self.weights_[0])
        self.coefficients_ = self.weights_[1:].copy()
        self.coef_ = self.coefficients_

        # Predict on the SAME training data we just fit on, purely to
        # measure how well the model explains it (this is a training-set R²,
        # not a held-out test score).
        y_pred: np.ndarray = X_b.dot(self.weights_)
        # "Residual" = leftover error. Squaring and summing every residual
        # gives the total squared error the model actually made.
        ss_res: float = float(np.sum((y - y_pred) ** 2))
        # The baseline to compare against is "always predict the average of
        # y" — this measures how much the true values vary around their own
        # mean, i.e. how much error that trivial baseline would make.
        ss_tot: float = float(np.sum((y - np.mean(y)) ** 2))
        # R² = 1 - (our error / baseline error). If y has no variance at all
        # (ss_tot == 0), this ratio would divide by zero, so that case is
        # special-cased to report a perfect 1.0 instead.
        self.r2score_ = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 1.0
        return self

    # Receives: X — new feature data to predict on, shape (n_samples,) or
    #   (n_samples, n_features).
    # Calculates: the model's linear prediction `w0 + w1*x1 + w2*x2 + ...`
    #   for every row of X, using the weights learned in fit().
    # Returns: an array of shape (n_samples,) — one predicted value per row
    #   of X.
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for the given input.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples,) or (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        # Same reshape-if-1D and bias-column trick used in fit(), so
        # predictions can be computed with the exact same matrix layout the
        # weights were learned against.
        if self.weights_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # One matrix multiply computes every sample's prediction at once:
        # row i of X_b dotted with `weights_` gives
        # w0*1 + w1*x1 + w2*x2 + ... for that sample.
        return X_b.dot(self.weights_)


class SGDRegression:
    """Linear regression via Mini-Batch Stochastic Gradient Descent.

    Gradient update rule based on Andrew Ng's CS229 Machine Learning
    course notes (Stanford, Spring 2022), Chapter 1 — Supervised Learning,
    Section 1.1 LMS algorithm:
      - p. 11  Batch gradient descent      (Eq. 1.1)
      - p. 13  Stochastic gradient descent (Eq. 1.2)
    Reference: https://cs229.stanford.edu/lectures-spring2022/main_notes.pdf
    Uses J = (1/2m)||Xw + b - y||² whose gradient w.r.t. w is (1/m) * Xᵀ(Xw + b - y)
    — the 1/2 in the cost cancels the factor of 2 from the derivative.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for each gradient update.
    n_iterations : int, default=1000
        Number of full passes over the training data (epochs).
    batch_size : int, default=32
        Number of samples per mini-batch.
    alpha : float, default=0.0
        L2 regularization strength added to the weight gradient (bias excluded).

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_features,)
        Feature weights. Set after fit().
    bias_ : float
        Intercept term w0. Set after fit().
    intercept_ : float
        Alias for bias_. Set after fit().
    coefficients_ : np.ndarray
        Alias for weights_. Set after fit().
    coef_ : np.ndarray
        Alias for weights_ (scikit-learn compatible). Set after fit().
    r2score_ : float
        R² score evaluated on the training data. Set after fit().
    loss_history_ : list of float
        Mean MSE per epoch — useful for checking convergence.
    """

    # Receives: learning_rate (how big a step to take on each gradient
    #   update), n_iterations (how many full passes over the data to run),
    #   batch_size (how many samples go into each mini-batch update), and
    #   alpha (L2 regularization strength).
    # Calculates: nothing yet — just remembers every hyperparameter and
    #   sets every fitted attribute to a placeholder (None or an empty
    #   list) until fit() actually trains the model.
    # Returns: nothing — __init__ only prepares the object.
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        batch_size: int = 32,
        alpha: float = 0.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.alpha = alpha
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.intercept_: Optional[float] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.r2score_: Optional[float] = None
        self.loss_history_: list = []

    # Receives: X — the (ideally pre-scaled) training feature matrix, and
    #   y — the true target value for each sample.
    # Calculates: repeatedly shuffles the data, splits it into mini-batches,
    #   and nudges `weights_`/`bias_` a small step in the direction that
    #   reduces the mean-squared-error on each batch, for `n_iterations`
    #   full passes over the data.
    # Returns: self — after storing `weights_`, `bias_`, the
    #   convenience aliases (`intercept_`, `coefficients_`, `coef_`),
    #   `r2score_`, and `loss_history_` as attributes on `self`.
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGDRegression":
        """Train using mini-batch gradient descent.

        Feature scaling (standardisation) is strongly recommended before
        calling fit, as gradient descent is sensitive to feature scale.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix (should be pre-scaled).
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        SGDRegression
            The fitted model (allows ``model.fit(X, y).predict(X)``).
        """
        # Unpack X's shape once: how many samples, and how many features
        # each one has.
        n_samples, n_features = X.shape
        # Start the weights at all-zeros and the bias at 0 — gradient
        # descent will nudge these towards good values over many iterations.
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        # One full pass over the entire training set is one "epoch"; we run
        # `n_iterations` of them.
        for _ in range(self.n_iterations):
            # shuffle each epoch to avoid systematic ordering bias.
            # np.random.permutation(n_samples) returns a random reordering of
            # the indices 0..n_samples-1; indexing X and y with it produces
            # the same dataset, just in a freshly randomised row order.
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            # We'll record this epoch's per-batch losses here, to average
            # into one number at the end of the epoch.
            batch_losses = []
            # Walk through the shuffled data in chunks of `batch_size` rows
            # at a time — each chunk is one "mini-batch".
            for start in range(0, n_samples, self.batch_size):
                # Python slicing like this is automatically safe near the
                # end of the array: if `start + batch_size` overshoots the
                # array's length, it just stops at the last row instead of
                # raising an error, so the very last batch of an epoch can
                # be smaller than `batch_size`.
                X_batch = X_shuffled[start : start + self.batch_size]
                y_batch = y_shuffled[start : start + self.batch_size]
                m = len(X_batch)

                # Gradient from Andrew Ng CS229 notes p. 11
                # https://cs229.stanford.edu/lectures-spring2022/main_notes.pdf
                # (1/m) * Xᵀ(Xw + b - y)
                # `X_batch.dot(self.weights_) + self.bias_` is this batch's
                # current predictions; subtracting `y_batch` gives the
                # per-sample error (how far off each prediction currently is).
                error = X_batch.dot(self.weights_) + self.bias_ - y_batch

                # The gradient tells us which direction increases the loss
                # the fastest; `X_batch.T.dot(error) / m` is the average,
                # per-feature contribution to that direction, and
                # `self.alpha * self.weights_` adds the Ridge penalty's own
                # gradient (proportional to the current weight itself).
                grad_w = (1.0 / m) * X_batch.T.dot(error) + self.alpha * self.weights_
                # The bias's gradient is simpler: just the average error
                # across the batch (no per-feature multiplication needed).
                grad_b = (1.0 / m) * np.sum(error)

                # Step the weights and bias a small distance (scaled by
                # `learning_rate`) in the OPPOSITE direction of the
                # gradient, since the gradient points toward increasing
                # loss and we want to decrease it.
                self.weights_ -= self.learning_rate * grad_w
                self.bias_    -= self.learning_rate * grad_b

                # Record this batch's own mean squared error, purely for the
                # loss_history_ log (doesn't affect training itself).
                batch_losses.append(float(np.mean(error ** 2)))

            # Average this epoch's batch-by-batch losses into a single
            # number and remember it, so `loss_history_` ends up with one
            # entry per epoch — handy for plotting a learning curve later.
            self.loss_history_.append(float(np.mean(batch_losses)))

        # convenience aliases.
        # Give the fitted weights/bias the same familiar sklearn-style names
        # used elsewhere in this library, all pointing at the exact same
        # values (not copies where it doesn't matter, `.copy()` only where
        # we don't want later code to accidentally mutate `weights_` through
        # the alias).
        self.intercept_ = self.bias_
        self.coefficients_ = self.weights_.copy()
        self.coef_ = self.coefficients_

        # Same R² computation as in LinearRegressor.fit — see that method
        # for the full walkthrough of what SS_res/SS_tot mean.
        y_pred: np.ndarray = X.dot(self.weights_) + self.bias_
        ss_res: float = float(np.sum((y - y_pred) ** 2))
        ss_tot: float = float(np.sum((y - np.mean(y)) ** 2))
        self.r2score_ = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 1.0
        return self

    # Receives: X — new (pre-scaled, using the same scaling as training)
    #   feature data to predict on.
    # Calculates: the linear prediction `X · weights_ + bias_` for every
    #   row of X.
    # Returns: an array of shape (n_samples,) — one predicted value per row
    #   of X.
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for the given input.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix (apply the same scaling used during fit).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        # Can't predict before fit() has learned the weights.
        if self.weights_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        # One matrix multiply plus the bias gives every sample's prediction
        # at once — no explicit per-sample loop needed.
        return X.dot(self.weights_) + self.bias_
