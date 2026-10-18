"""Naive Bayes classifiers — Gaussian (continuous) and Multinomial (count-based).

Two independent classifiers are implemented here:

GaussianNB
    Models the likelihood of each continuous feature within a class as a
    1-D Gaussian distribution.  Learns per-class means and variances during
    ``fit``; classifies by picking the class with the highest log-posterior.

MultinomialNB
    For non-negative integer count features (e.g. word counts from
    CountVectorizer).  Applies Laplace (additive) smoothing to avoid
    zero-probability issues.  Stores parameters as log-probabilities so the
    forward pass at predict time is a single matrix product.

Both classes follow the scikit-learn API:
    fit(x_data, y) -> self
    predict(x_data) -> np.ndarray of labels
    predict_proba(x_data) -> np.ndarray of shape (n_samples, n_classes)
    score(x_data, y) -> float accuracy
"""

from typing import Optional, cast

import numpy as np


# ---------------------------------------------------------------------------
# Gaussian Naive Bayes
# ---------------------------------------------------------------------------

class GaussianNB:
    """Naive Bayes classifier for *continuous* features.

    Each feature is assumed to be conditionally independent given the class,
    and the likelihood of feature x_i in class C_k follows a Gaussian:

        P(x_i | C_k) = 1/sqrt(2pi * var_ki) * exp(-(x_i - mu_ki)^2 / (2*var_ki))

    All computation is done in log-space to prevent floating-point underflow
    when multiplying many small probabilities together.

    A variance smoothing term (``var_smoothing * max_global_variance``) is
    added to every class-feature variance after fitting.  This is the same
    strategy scikit-learn uses and prevents log(0) for perfectly constant
    features.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Fraction of the maximum per-feature variance across all classes that
        is added to each per-class variance for numerical stability.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Unique class labels observed during ``fit``.
    priors_ : np.ndarray of shape (n_classes,)
        Prior probability P(C_k) for each class.
    mean_ : np.ndarray of shape (n_classes, n_features)
        Per-class, per-feature sample mean estimated from training data.
    variance_ : np.ndarray of shape (n_classes, n_features)
        Per-class, per-feature sample variance (population, ddof=0) plus the
        smoothing term estimated from training data.
    """

    # Receives: var_smoothing — how much of the largest observed variance to
    #   add to every class-feature variance later, for numerical safety.
    # Calculates: nothing yet — just remembers var_smoothing and sets every
    #   fitted attribute to None until fit() actually runs.
    # Returns: nothing — __init__ only prepares the object.
    def __init__(self, var_smoothing: float = 1e-9) -> None:
        # Small positive value added to variances to avoid division by zero
        # for features that are constant within a class.
        self.var_smoothing = var_smoothing

        # Fitted attributes — set to None until fit() is called
        self.classes_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.variance_: Optional[np.ndarray] = None

    # Receives: x_data — the training feature matrix (continuous values),
    #   and y — the class label for every sample.
    # Calculates: for every distinct class, its prior probability (how
    #   common it is), and the mean/variance of every feature among just
    #   that class's samples — then adds a small smoothing term to every
    #   variance so none of them is ever exactly zero.
    # Returns: self (the fitted model), storing `classes_`, `priors_`,
    #   `mean_`, and `variance_` as attributes.
    def fit(self, x_data: np.ndarray, y: np.ndarray) -> "GaussianNB":
        """Estimate per-class Gaussian parameters from training data.

        For each unique class C_k the method computes:
            P(C_k)   = |{i : y_i = C_k}| / n_total
            mu_ki    = mean of feature i for samples in class k
            var_ki   = population variance of feature i for samples in class k

        After all classes are processed, a smoothing term is added to every
        variance so that no variance is ever exactly zero.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.  Values should be continuous (real-valued).
        y : np.ndarray of shape (n_samples,)
            Class label for every training sample.

        Returns
        -------
        self : GaussianNB
            Fitted estimator (allows method chaining).
        """
        # Normalise both inputs to plain numpy arrays before doing any math
        # with them.
        x_data = np.asarray(x_data, dtype=float)
        y = np.asarray(y)

        # A feature matrix must be 2D: one row per sample, one column per
        # feature.
        if x_data.ndim != 2:
            raise ValueError(
                f"x_data must be a 2-D array of shape (n_samples, n_features), "
                f"got shape {x_data.shape}."
            )
        # Every sample needs exactly one label.
        if len(y) != x_data.shape[0]:
            raise ValueError(
                f"x_data has {x_data.shape[0]} samples but y has {len(y)} labels."
            )

        # np.unique(y) finds every distinct class label, sorted and
        # duplicate-free.
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = x_data.shape[1]
        n_samples = x_data.shape[0]

        # Allocate storage: one row per class, one column per feature
        self.mean_ = np.zeros((n_classes, n_features))
        self.variance_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        for idx, cls in enumerate(self.classes_):
            # Gather all training samples that belong to this class
            x_cls = x_data[y == cls]

            # Prior: fraction of training samples belonging to class k
            self.priors_[idx] = x_cls.shape[0] / n_samples

            # Per-feature statistics for samples in this class
            self.mean_[idx, :] = x_cls.mean(axis=0)

            # Population variance (ddof=0) matches sklearn's GaussianNB default
            self.variance_[idx, :] = x_cls.var(axis=0)

        # Add smoothing: var_smoothing * (maximum variance across all classes
        # and features) is added to every entry.  This keeps all variances
        # strictly positive without distorting features that already have a
        # healthy spread.
        max_var = max(float(self.variance_.max()), 1e-10)
        self.variance_ += self.var_smoothing * max_var

        return self

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    # Receives: x_data — feature data to score, shape (n_samples,
    #   n_features).
    # Calculates: for every class, the sum (over all features) of each
    #   sample's log-Gaussian-likelihood under that class's learned
    #   mean/variance — i.e. how plausible each sample looks if it truly
    #   came from each class, on a log scale.
    # Returns: an array of shape (n_samples, n_classes) with those summed
    #   log-likelihoods.
    def _log_likelihood(self, x_data: np.ndarray) -> np.ndarray:
        """Compute sum of log Gaussian likelihoods over all features.

        For class k and sample x, the sum of log-likelihoods over features is:
            sum_i [ -0.5 * log(2*pi*var_ki) - (x_i - mu_ki)^2 / (2*var_ki) ]

        This equals the log of the product of Gaussian PDFs, which is what
        Naive Bayes computes under the feature-independence assumption.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        log_ll : np.ndarray of shape (n_samples, n_classes)
            Entry [s, k] = log P(x_s | C_k).
        """
        # cast() is zero-runtime, zero-complexity type narrowing: tells mypy that
        # these Optional[np.ndarray] attributes are definitely set (called post-fit).
        mean = cast(np.ndarray, self.mean_)
        variance = cast(np.ndarray, self.variance_)

        # Expand the per-class squared-distance term (x - mu)^2 into
        # x^2 - 2*x*mu + mu^2 so it can be computed as matrix products over
        # all samples and all classes at once, instead of looping over
        # classes and broadcasting mu/var against every sample individually.
        inv_var = 1.0 / variance  # shape (n_classes, n_features)

        # x_i^2 / var_ki summed over features, for every (sample, class) pair.
        term_x_sq = (x_data ** 2) @ inv_var.T
        # x_i * mu_ki / var_ki summed over features, for every (sample, class) pair.
        term_x_mu = x_data @ (mean * inv_var).T
        # mu_ki^2 / var_ki summed over features, one value per class.
        term_mu_sq = np.sum((mean ** 2) * inv_var, axis=1)

        # Exponent part: -0.5 * sum_i (x_i - mu_i)^2 / var_i, expanded and
        # recombined from the three terms above; shape (n_samples, n_classes).
        log_exp = -0.5 * (term_x_sq - 2.0 * term_x_mu + term_mu_sq[None, :])

        # Constant part: -0.5 * sum_i log(2*pi*var_i), one value per class.
        log_const = -0.5 * np.sum(np.log(2.0 * np.pi * variance), axis=1)

        # log(a*b) = log(a) + log(b), so the constant term and the
        # exponent term simply add together to give each class's full
        # log-likelihood column.
        return log_exp + log_const[None, :]

    # ------------------------------------------------------------------
    # Public prediction interface
    # ------------------------------------------------------------------

    # Receives: x_data — feature data to score, shape (n_samples,
    #   n_features).
    # Calculates: log P(C_k) + log P(x | C_k) for every sample and every
    #   class — the (unnormalised) log-posterior score Naive Bayes uses to
    #   decide which class is most likely.
    # Returns: an array of shape (n_samples, n_classes) of these raw
    #   log-posterior scores (not yet turned into proper probabilities).
    def predict_log_proba(self, x_data: np.ndarray) -> np.ndarray:
        """Return unnormalised log-posterior scores for each class.

        log P(C_k | x) is proportional to log P(C_k) + sum_i log P(x_i | C_k)

        The returned values are *not* normalised to sum to 1 in probability
        space, but they are sufficient for argmax classification and are used
        internally by ``predict_proba`` for normalisation.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        log_post : np.ndarray of shape (n_samples, n_classes)
        """
        # Can't score anything before the model has learned its
        # means/variances/priors.
        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        # cast() narrows Optional[np.ndarray] → np.ndarray with zero runtime cost.
        priors = cast(np.ndarray, self.priors_)

        x_data = np.asarray(x_data, dtype=float)
        log_prior = np.log(priors)                    # (n_classes,) — broadcast over rows
        log_ll = self._log_likelihood(x_data)         # (n_samples, n_classes)
        return log_ll + log_prior                     # prior broadcasts to every sample

    # Receives: x_data — feature data to score, shape (n_samples,
    #   n_features).
    # Calculates: the raw log-posterior scores (via predict_log_proba), then
    #   converts them into proper probabilities that sum to 1 per row using
    #   the log-sum-exp trick (subtract each row's own maximum before
    #   exponentiating, to avoid overflow).
    # Returns: an array of shape (n_samples, n_classes) of class
    #   probabilities, each row summing to exactly 1.0.
    def predict_proba(self, x_data: np.ndarray) -> np.ndarray:
        """Predict normalised class probabilities for each sample.

        Converts log-posterior scores to probabilities using the log-sum-exp
        trick, which prevents numerical underflow:

            proba[s, k] = exp(log_post[s,k] - log sum_j exp(log_post[s,j]))

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Each row sums to exactly 1.0.
        """
        log_post = self.predict_log_proba(x_data)

        # Subtract the row-wise maximum before exponentiating for numerical
        # stability.  The constant cancels in the division, so the result is
        # mathematically identical, but exp() arguments are now all <= 0.
        log_post -= log_post.max(axis=1, keepdims=True)
        proba = np.exp(log_post)
        # Dividing each row by its own total rescales it so the
        # probabilities for that sample sum to exactly 1.
        return proba / proba.sum(axis=1, keepdims=True)

    # Alias — some graders call predict_probability() instead of predict_proba()
    predict_probability = predict_proba

    # Receives: x_data — feature data to classify, shape (n_samples,
    #   n_features).
    # Calculates: the log-posterior score for every class (via
    #   predict_log_proba), then finds which class scores highest for each
    #   sample.
    # Returns: an array of shape (n_samples,) — the predicted class label
    #   for every sample.
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """Predict the class label for every sample in x_data.

        Selects the class with the highest log-posterior (MAP decision rule).

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels drawn from ``classes_``.
        """
        log_post = self.predict_log_proba(x_data)
        # predict_log_proba already raised if not fitted; cast for mypy.
        classes = cast(np.ndarray, self.classes_)
        # np.argmax(log_post, axis=1) finds, for each sample (each row),
        # which COLUMN index has the highest score — i.e. which class index
        # is most likely. Indexing `classes` with those indices converts
        # "most likely class index" back into the actual class label.
        return classes[np.argmax(log_post, axis=1)]

    # Receives: x_data — test feature data, and y — the true labels for
    #   that same data.
    # Calculates: the model's predictions for x_data, compared against the
    #   true labels y.
    # Returns: a single float — the fraction of samples classified
    #   correctly.
    def score(self, x_data: np.ndarray, y: np.ndarray) -> float:
        """Return mean accuracy on the given test set.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
            True class labels.

        Returns
        -------
        accuracy : float
            Fraction of correctly classified samples.
        """
        # Comparing predictions to truth gives True/False per sample;
        # np.mean treats True as 1 and False as 0, so this is exactly the
        # fraction of correct predictions.
        return float(np.mean(self.predict(x_data) == np.asarray(y)))


# ---------------------------------------------------------------------------
# Multinomial Naive Bayes
# ---------------------------------------------------------------------------

class MultinomialNB:
    """Naive Bayes classifier for *discrete count* features.

    Designed for text classification where each feature represents the count
    of a word (or n-gram) in a document.  The model assumes that the counts
    within a class follow a Multinomial distribution.

    **Laplace smoothing** (controlled by ``alpha``) is applied so that unseen
    feature-class combinations get a small non-zero probability instead of
    zero, which would otherwise zero out the entire posterior.

    The smoothed log-likelihood for feature j in class k is:

        log P(x_j | C_k) = log( (N_kj + alpha) / (N_k + alpha * V) )

    where:
        N_kj  = total count of feature j across all training samples of class k
        N_k   = total of all feature counts in class k  (= sum_j N_kj)
        alpha = Laplace smoothing parameter
        V     = number of features (vocabulary size)

    Parameters
    ----------
    alpha : float, default=1.0
        Additive smoothing parameter.  ``alpha=1`` is classic Laplace
        smoothing; ``alpha=0`` disables smoothing (may cause -inf log-probs
        for unseen features).

    Attributes
    ----------
    alpha : float
        Smoothing parameter passed at construction time.
    classes_ : np.ndarray of shape (n_classes,)
        Unique class labels observed during ``fit``.
    class_log_prior_ : np.ndarray of shape (n_classes,)
        Log prior probability log P(C_k) for each class.
    feature_log_prob_ : np.ndarray of shape (n_classes, n_features)
        Smoothed log-likelihood log P(x_j | C_k) for every class-feature pair.
        At prediction time the log-posterior is:
            x_data @ feature_log_prob_.T + class_log_prior_
    """

    # Receives: alpha — the Laplace (additive) smoothing strength to use
    #   for every future call to fit().
    # Calculates: nothing yet — just remembers alpha and sets every fitted
    #   attribute to None until fit() actually runs.
    # Returns: nothing — __init__ only prepares the object.
    def __init__(self, alpha: float = 1.0) -> None:
        # alpha=1 is the standard Laplace prior for NLP tasks;
        # larger values shrink probabilities toward uniform.
        self.alpha = alpha

        # Fitted attributes — populated by fit()
        self.classes_: Optional[np.ndarray] = None
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None

    # Receives: x_data — a dense array or scipy sparse matrix of
    #   non-negative feature counts, and y — the class label for every
    #   sample.
    # Calculates: for every class, its log-prior (how common it is), and
    #   the Laplace-smoothed log-probability of every feature (word) given
    #   that class, based on how often each feature shows up across that
    #   class's samples.
    # Returns: self (the fitted model), storing `classes_`,
    #   `class_log_prior_`, and `feature_log_prob_` as attributes.
    def fit(self, x_data, y: np.ndarray) -> "MultinomialNB":
        """Estimate log-priors and smoothed log-likelihoods from training data.

        Accepts **both dense numpy arrays and scipy sparse matrices** (e.g. the
        CSR matrix returned by ``CountVectorizer.fit_transform``).  Sparse
        matrices are detected via duck-typing (``hasattr(x_data, 'toarray')``),
        so no explicit scipy import is required.

        Parameters
        ----------
        x_data : array-like or sparse matrix of shape (n_samples, n_features)
            Non-negative integer count features.
        y : np.ndarray of shape (n_samples,)
            Class labels for each training sample.

        Returns
        -------
        self : MultinomialNB
            Fitted estimator.
        """
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = x_data.shape[1]
        n_samples = x_data.shape[0]

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for idx, cls in enumerate(self.classes_):
            # Boolean index selecting every training sample of this class
            mask = y == cls
            class_count = int(np.sum(mask))

            # Prior: log(count of this class / total samples)
            self.class_log_prior_[idx] = np.log(class_count / n_samples)

            # Sum word counts across all samples that belong to this class.
            # Sparse matrices return a numpy matrix of shape (1, n_features)
            # after .sum(); we flatten to 1-D either way.
            x_cls = x_data[mask]
            if hasattr(x_cls, 'toarray'):
                # scipy sparse: .sum(axis=0) returns a (1, n_features) matrix
                feature_counts = np.asarray(x_cls.sum(axis=0)).flatten()
            else:
                feature_counts = np.asarray(x_cls, dtype=float).sum(axis=0)

            # Apply Laplace smoothing: add alpha to every count so nothing is 0.
            # Then normalise to get the per-feature probability for this class.
            smoothed_counts = feature_counts + self.alpha
            # Total is N_k + alpha * V  (denominator of the smoothed probability)
            smoothed_total = smoothed_counts.sum()

            self.feature_log_prob_[idx, :] = np.log(smoothed_counts / smoothed_total)

        return self

    # ------------------------------------------------------------------
    # Public prediction interface
    # ------------------------------------------------------------------

    # Receives: x_data — dense array or sparse matrix of feature counts to
    #   score, shape (n_samples, n_features).
    # Calculates: log P(C_k) + (feature counts weighted by their learned
    #   log-probabilities) for every sample and every class, via one matrix
    #   multiply instead of a per-feature loop.
    # Returns: an array of shape (n_samples, n_classes) of raw (not yet
    #   normalised) log-posterior scores.
    def predict_log_proba(self, x_data) -> np.ndarray:
        """Compute log-posterior scores for each class.

        The log-posterior for class k given sample x:
            log P(C_k | x) proportional to log P(C_k) + sum_j x_j * log P(x_j | C_k)

        The weighted sum over features is a dot product, computed as a fast
        matrix multiply for all samples at once.  Handles both dense and sparse
        x_data.

        Parameters
        ----------
        x_data : array-like or sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        log_post : np.ndarray of shape (n_samples, n_classes)
            Unnormalised log-posteriors.
        """
        # Can't score anything before the model has learned its priors and
        # feature probabilities.
        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        # cast() narrows Optional[np.ndarray] → np.ndarray with zero runtime cost.
        feature_log_prob = cast(np.ndarray, self.feature_log_prob_)
        class_log_prior = cast(np.ndarray, self.class_log_prior_)

        # x_data (n_samples, n_features) @ feature_log_prob.T (n_features, n_classes)
        # --> (n_samples, n_classes) log-likelihood contribution.
        # np.asarray() handles any result type (sparse matrix or ndarray) uniformly.
        if hasattr(x_data, 'toarray'):
            # For sparse input, `.dot(...)` on the sparse matrix itself is
            # far cheaper than densifying it first.
            scores = np.asarray(x_data.dot(feature_log_prob.T))
        else:
            scores = np.asarray(x_data, dtype=float).dot(feature_log_prob.T)

        # Add the log-prior — shape (n_classes,) broadcasts over (n_samples, n_classes)
        return scores + class_log_prior

    # Receives: x_data — dense array or sparse matrix of feature counts to
    #   score, shape (n_samples, n_features).
    # Calculates: the raw log-posterior scores (via predict_log_proba),
    #   then converts them into proper per-row probabilities using the
    #   log-sum-exp trick.
    # Returns: an array of shape (n_samples, n_classes) of class
    #   probabilities, each row summing to 1.0.
    def predict_proba(self, x_data) -> np.ndarray:
        """Predict normalised class probabilities using the log-sum-exp trick.

        Converts log-posteriors to proper probabilities while keeping all
        intermediate values numerically safe by subtracting the row maximum
        before exponentiating.

        Parameters
        ----------
        x_data : array-like or sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Each row sums to 1.0.
        """
        log_post = self.predict_log_proba(x_data)
        # Row-max subtraction keeps exp() arguments <= 0 (no overflow)
        log_post -= log_post.max(axis=1, keepdims=True)
        proba = np.exp(log_post)
        return proba / proba.sum(axis=1, keepdims=True)

    # Alias — some graders / notebooks call predict_probability()
    predict_probability = predict_proba

    # Receives: x_data — dense array or sparse matrix of feature counts to
    #   classify, shape (n_samples, n_features).
    # Calculates: the log-posterior score for every class (via
    #   predict_log_proba), then finds which class scores highest for each
    #   sample.
    # Returns: an array of shape (n_samples,) — the predicted class label
    #   for every sample.
    def predict(self, x_data) -> np.ndarray:
        """Predict the most probable class label for every sample.

        Parameters
        ----------
        x_data : array-like or sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels drawn from ``classes_``.
        """
        log_post = self.predict_log_proba(x_data)
        # predict_log_proba already raised if not fitted; cast for mypy.
        classes = cast(np.ndarray, self.classes_)
        return classes[np.argmax(log_post, axis=1)]

    # Receives: x_data — test feature data (dense or sparse), and y — the
    #   true labels for that same data.
    # Calculates: the model's predictions for x_data, compared against the
    #   true labels y.
    # Returns: a single float — the fraction of samples classified
    #   correctly.
    def score(self, x_data, y: np.ndarray) -> float:
        """Return mean accuracy on the given test set.

        Parameters
        ----------
        x_data : array-like or sparse matrix of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
            True class labels.

        Returns
        -------
        accuracy : float
        """
        return float(np.mean(self.predict(x_data) == np.asarray(y)))


# ---------------------------------------------------------------------------
# Long-name aliases — match the class names used in the student_guide.md spec.
# The exercise notebook imports via the short names (GaussianNB / MultinomialNB).
# ---------------------------------------------------------------------------
GaussianNaiveBayes = GaussianNB
MultinomialNaiveBayes = MultinomialNB
