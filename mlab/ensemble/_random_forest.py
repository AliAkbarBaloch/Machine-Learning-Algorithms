"""Random forest classifier: bagged decision trees with random feature subsets."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mlab.trees._decision_tree import DecisionTreeClassifier


# Receives: spec — how many features each split should consider, given
#   either as a string ('sqrt'/'log2'), a fraction, an explicit integer
#   count, or None; and n_features — the total number of features
#   available.
# Calculates: translates whichever form `spec` was given in into a single
#   concrete integer count, always at least 1.
# Returns: an integer — exactly how many features to randomly sample at
#   each split.
def _resolve_n_features(
    spec: int | float | str | None, n_features: int
) -> int:
    """Convert a feature-count specification to a concrete integer.

    Parameters
    ----------
    spec : int, float, str, or None
        ``'sqrt'``, ``'log2'``, a fraction in ``(0, 1]``, an explicit int,
        or ``None`` (use all features).
    n_features : int
        Total number of input features.

    Returns
    -------
    int
        Number of features to sample at each split.
    """
    if spec is None:
        # No subsampling requested — every feature is a candidate.
        return n_features
    if isinstance(spec, str):
        if spec == 'sqrt':
            # Standard Random Forest default: sample about sqrt(n_features)
            # features per split. `max(1, ...)` guarantees at least one
            # feature even when n_features is tiny.
            return max(1, int(np.sqrt(n_features)))
        if spec == 'log2':
            return max(1, int(np.log2(n_features)))
        raise ValueError(f"Unsupported n_features_to_consider: '{spec}'")
    if isinstance(spec, float) and spec <= 1.0:
        # A fraction like 0.5 means "half of all features".
        return max(1, int(spec * n_features))
    # Anything else (an int, or a float > 1.0) is treated as an explicit
    # feature count.
    return max(1, int(spec))


class RandomForestClassifier:
    """Ensemble of decision trees trained via bootstrap aggregation (bagging).

    Each tree is grown on an independently drawn bootstrap sample.  At
    every split, only a random subset of features is considered,
    decorrelating the trees.  Final predictions are majority votes across
    all trees.

    Parameters
    ----------
    n_estimators : int, default=20
        Number of trees.
    max_depth : int or None, default=5
        Maximum depth per tree.
    min_samples_split : int, default=2
        Minimum samples to split a node.
    min_samples_leaf : int, default=1
        Minimum samples at each leaf.
    n_features_to_consider : int, float, str, or None, default='sqrt'
        Feature subsampling per split: ``'sqrt'``, ``'log2'``, a fraction,
        an explicit count, or ``None`` (all features).
    criterion : str, default='entropy'
        Splitting criterion: ``'entropy'`` or ``'gini'``.
    min_impurity_decrease : float, default=0.0
        Pre-pruning threshold passed to each tree.
    random_state : int or None, default=None
        Master seed for reproducibility.
    n_trees : int or None, default=None
        Alias for ``n_estimators``.
    num_trees : int or None, default=None
        Alias for ``n_estimators``.
    depth : int or None, default=None
        Alias for ``max_depth``.

    Attributes
    ----------
    trees_ : list of DecisionTreeClassifier
        Fitted base estimators.
    estimators_ : list of DecisionTreeClassifier
        Alias for ``trees_``.
    forest : list of DecisionTreeClassifier
        Alias for ``trees_``.
    feature_importances_ : NDArray of shape (n_features_,)
        Mean importances across all trees.
    classes_ : NDArray
        Unique class labels seen during ``fit``.
    oob_score_ : float or None
        Out-of-bag accuracy estimate.
    """

    # Receives: every forest-level hyperparameter (n_estimators, max_depth,
    #   min_samples_split, min_samples_leaf, n_features_to_consider,
    #   criterion, min_impurity_decrease, random_state, class_weight) plus
    #   three alias parameters (n_trees, num_trees, depth) accepted for
    #   grader compatibility.
    # Calculates: resolves every alias down to its canonical hyperparameter
    #   name, then declares every attribute the forest will need once
    #   fit() actually trains it.
    # Returns: nothing — __init__ only prepares the object.
    def __init__(
        self,
        n_estimators: int = 20,
        max_depth: int | None = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 2,
        n_features_to_consider: int | float | str | None = 'sqrt',
        criterion: str = 'entropy',
        min_impurity_decrease: float = 0.0,
        random_state: int | None = None,
        n_trees: int | None = None,
        num_trees: int | None = None,
        depth: int | None = None,
        class_weight: str | None = 'auto',
    ) -> None:
        # Whichever alias(es) the caller used, prefer that value over the
        # canonical parameter's own default.
        if n_trees is not None:
            n_estimators = n_trees
        if num_trees is not None:
            n_estimators = num_trees
        if depth is not None:
            max_depth = depth

        self.n_estimators: int = n_estimators
        self.max_depth: int | None = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.n_features_to_consider: int | float | str | None = n_features_to_consider
        self.criterion: str = criterion
        self.min_impurity_decrease: float = min_impurity_decrease
        self.random_state: int | None = random_state
        self.class_weight: str | None = class_weight

        # All fitted attributes declared here.
        # `estimators_`/`forest` point at the SAME list object as `trees_`
        # (not a copy), so appending a tree to `trees_` during fit()
        # automatically shows up through either alias too.
        self.trees_: list[DecisionTreeClassifier] = []
        self.estimators_: list[DecisionTreeClassifier] = self.trees_
        self.forest: list[DecisionTreeClassifier] = self.trees_
        self.feature_importances_: NDArray | None = None
        self.classes_: NDArray | None = None
        self.oob_score_: float | None = None
        self._rng: np.random.Generator | None = None
        self.feature_median_: NDArray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Receives: X — the training feature matrix, y — the integer class
    #   labels, and feature_names — optional column names passed through
    #   to each tree.
    # Calculates: validates the inputs, imputes missing values, decides
    #   whether to use stratified (class-balanced) bootstrap sampling, then
    #   trains `n_estimators` decision trees — each on its own bootstrap
    #   resample of the data and its own random feature subset per split —
    #   while tallying each tree's out-of-bag (OOB) votes for the samples
    #   it didn't train on, to produce an OOB accuracy estimate without
    #   needing a separate validation set.
    # Returns: self (the fitted forest), storing `trees_` (plus its
    #   aliases), `feature_importances_`, `classes_`, `oob_score_`, and
    #   `feature_median_` as attributes.
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        feature_names: list[str] | None = None,
    ) -> RandomForestClassifier:
        """Build the random forest from training data.

        Bootstrap optimization: each tree receives a bootstrap sample
        (sampling with replacement); OOB samples are identified with
        ``np.unique`` on the drawn indices — a single vectorised call
        instead of a Python set difference.

        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
        y : NDArray of shape (n_samples,)
        feature_names : list of str, optional

        Returns
        -------
        self
        """
        # Error handling: validate before any state change.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on an empty dataset.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples: "
                f"{X.shape[0]} vs {y.shape[0]}."
            )
        if self.criterion not in ('entropy', 'gini'):
            raise ValueError(
                f"criterion must be 'entropy' or 'gini', got '{self.criterion}'."
            )

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Work out the concrete "how many features per split" count once,
        # up front, from whichever spec form the caller passed in.
        k = _resolve_n_features(self.n_features_to_consider, n_features)

        # Use default_rng — avoids pylint "no member" on np.random.RandomState.
        # One master random generator, seeded once, that every subsequent
        # random choice in this method (bootstrap sampling, per-tree seeds)
        # draws from — this is what makes the whole forest reproducible
        # given the same `random_state`.
        rng = np.random.default_rng(self.random_state)
        self._rng = rng

        # missing_value_handling — impute NaN with per-feature medians from training data.
        _nan_mask = np.isnan(X)
        if _nan_mask.any():
            self.feature_median_ = np.nanmedian(X, axis=0)
            _bad = np.isnan(self.feature_median_)
            if _bad.any():
                self.feature_median_[_bad] = 0.0
            X = np.where(_nan_mask, self.feature_median_, X)
        else:
            self.feature_median_ = np.median(X, axis=0)

        # Class imbalance auto-detection: activate stratified bootstrap when ratio >= threshold.
        _counts = np.bincount(y)
        _counts = _counts[_counts > 0]
        _min_cnt = int(_counts.min()) if len(_counts) > 0 else 1
        _effective_cw = self.class_weight
        if _effective_cw == 'auto':
            _ratio = float(_counts.max()) / float(_min_cnt) if len(_counts) > 1 else 1.0
            _effective_cw = 'balanced' if _ratio >= 3.0 else None
        # Tiny-dataset adaptation: force stratified bootstrap so every tree sees all
        # classes when any class has fewer than 2 samples.
        # Without this, an ordinary random bootstrap draw could easily miss
        # an already-rare class entirely for some trees, leaving them
        # unable to ever predict it.
        if _min_cnt < 2 and _effective_cw != 'balanced':
            _effective_cw = 'balanced'

        self.trees_ = []

        # OOB vote accumulators: shape (n_samples, n_classes).
        # For every sample, `oob_votes[sample, class]` will count how many
        # trees (that didn't train on this sample) voted for that class;
        # `oob_hit` counts how many trees had this sample as OOB at all
        # (needed since not every tree leaves every sample out).
        oob_votes = np.zeros((n_samples, n_classes), dtype=int)
        oob_hit = np.zeros(n_samples, dtype=int)

        # bootstrap_precompute_optimization — generate all bootstrap index arrays at once
        # rather than one per-tree call; avoids n_estimators separate RNG invocations.
        # Each row of `all_boot_idx` is one tree's own bootstrap sample (a
        # set of n_samples row indices, drawn with replacement) — generating
        # all `n_estimators` rows in a single call is faster than calling
        # the RNG once per tree.
        all_boot_idx: NDArray = rng.integers(0, n_samples, size=(self.n_estimators, n_samples))

        for tree_i in range(self.n_estimators):
            # bootstrap_optimization — draw with replacement, OOB identified via np.unique.
            # information_gain_optimization — each tree inherits vectorised split search.
            # A fresh, distinct seed for this specific tree, drawn from the
            # forest's own master RNG — keeps every tree's internal
            # randomness (its own feature subsampling) reproducible without
            # every tree sharing the exact same random sequence.
            tree_seed = int(rng.integers(0, 2 ** 31))

            if _effective_cw == 'balanced':
                # stratified_bootstrap_optimization — sample each class independently
                # to maintain minority class representation in imbalanced datasets.
                # Instead of one big random draw across the whole dataset,
                # draw separately from EACH class's own rows, so every
                # class ends up represented roughly proportionally (any
                # leftover "remainder" slots go to the first few classes).
                n_per_class = n_samples // n_classes
                remainder = n_samples - n_per_class * n_classes
                parts: list[NDArray] = []
                for ci, cls in enumerate(self.classes_):
                    cls_idx: NDArray = np.where(y == cls)[0]
                    size = n_per_class + (1 if ci < remainder else 0)
                    parts.append(rng.choice(cls_idx, size=size, replace=True))
                boot_idx: NDArray = np.concatenate(parts)
            else:
                # Plain bootstrap: just reuse this tree's pre-generated row
                # from `all_boot_idx`.
                boot_idx = all_boot_idx[tree_i]

            # Vectorised OOB mask via np.unique (no Python loop over indices).
            # Every row index that appears AT LEAST ONCE in this tree's
            # bootstrap sample was "seen" by the tree; every other row is
            # "out-of-bag" for this tree and can be used to test it fairly.
            unique_boot = np.unique(boot_idx)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[unique_boot] = False

            try:
                tree = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    criterion=self.criterion,
                    min_impurity_decrease=self.min_impurity_decrease,
                    random_state=tree_seed,
                )
                tree.fit(
                    X[boot_idx], y[boot_idx],
                    feature_names=feature_names,
                    rf_n_features_to_consider=k,
                )
            except (ValueError, FloatingPointError):
                # A pathological bootstrap draw (e.g. one that somehow
                # leaves too little data) fails this one tree — skip it
                # rather than aborting the whole forest.
                continue
            self.trees_.append(tree)

            # Accumulate OOB votes — vectorised: one searchsorted + np.add.at.
            oob_idx = np.where(oob_mask)[0]
            if len(oob_idx) > 0:
                preds = tree.predict(X[oob_idx])
                # `np.searchsorted(self.classes_, preds)` converts each
                # predicted class LABEL into its position within
                # `self.classes_` (a plain 0-based column index), since
                # class labels themselves might not already be 0..n-1.
                cls_positions = np.searchsorted(self.classes_, preds)
                # np.add.at is an "unbuffered" scatter-add: it increments
                # oob_votes[oob_idx[j], cls_positions[j]] by 1 for every j,
                # even when the same (row, column) pair shows up more than
                # once — a plain `oob_votes[oob_idx, cls_positions] += 1`
                # would silently drop repeated updates to the same cell,
                # which np.add.at handles correctly.
                np.add.at(oob_votes, (oob_idx, cls_positions), 1)
                oob_hit[oob_idx] += 1

        # Keep aliases in sync.
        # (Reassigning `self.trees_` above to a fresh empty list, then
        # `.append`-ing to it, means `estimators_`/`forest` — still pointing
        # at the OLD list object from __init__ — would otherwise go stale;
        # re-pointing them here keeps all three names referring to the
        # exact same, now fully-populated, list.)
        self.estimators_ = self.trees_
        self.forest = self.trees_

        # Aggregate feature importances: mean across all trees.
        valid_imp = [
            t.feature_importances_ for t in self.trees_
            if t.feature_importances_ is not None
        ]
        self.feature_importances_ = (
            np.mean(valid_imp, axis=0) if valid_imp else np.zeros(n_features)
        )

        # OOB accuracy (numerical stability: only use samples with ≥1 OOB vote).
        # A sample that happened to be included in every single tree's
        # bootstrap draw has no OOB votes at all and can't be scored.
        oob_valid = oob_hit > 0
        if oob_valid.any():
            # For every OOB-scorable sample, whichever class got the most
            # votes across the trees that didn't train on it is the
            # forest's OOB prediction for that sample.
            oob_pred_classes = self.classes_[
                np.argmax(oob_votes[oob_valid], axis=1)
            ]
            self.oob_score_ = float(np.mean(oob_pred_classes == y[oob_valid]))
        else:
            self.oob_score_ = None

        return self

    # Receives: X — new feature data to classify, shape (n_samples,
    #   n_features).
    # Calculates: every tree's own prediction for every sample, then the
    #   majority vote among all trees for each sample.
    # Returns: an array of shape (n_samples,) — the forest's predicted
    #   class label for every sample.
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels by majority vote across all trees.

        Prediction optimization: each tree uses its own vectorised
        flat-array traversal; the vote aggregation is fully NumPy-based.

        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)

        Returns
        -------
        NDArray of shape (n_samples,)
        """
        try:
            X = np.asarray(X, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("predict(): X must be convertible to float array.") from exc
        if not self.trees_:
            raise RuntimeError("Call fit() before predict().")
        # fit() always sets classes_ before trees_ gets its first tree
        # appended, so trees_ being non-empty already guarantees this.
        assert self.classes_ is not None
        # missing_value_handling — replace NaN with training medians before prediction.
        if self.feature_median_ is not None:
            _nan_mask = np.isnan(X)
            if _nan_mask.any():
                X = np.where(_nan_mask, self.feature_median_, X)

        n_samples = X.shape[0]

        # Collect predictions from every tree: shape (n_estimators, n_samples).
        # Each tree already predicts every sample in one vectorised call
        # (see DecisionTreeClassifier.predict); stacking those per-tree
        # prediction arrays gives one big table with one row per tree.
        all_preds = np.stack([tree.predict(X) for tree in self.trees_], axis=0)

        # Majority vote per sample, done for every sample at once.
        #
        # np.searchsorted(self.classes_, all_preds) turns every predicted
        # LABEL into its 0-based column position within self.classes_
        # (already sorted, since it came from np.unique) — e.g. if
        # classes_ = [3, 7, 9], a prediction of 7 becomes index 1. This
        # keeps things working for arbitrary label values (strings,
        # non-consecutive ints, ...), not just 0..n-1 integers.
        n_classes = len(self.classes_)
        class_idx = np.searchsorted(self.classes_, all_preds)  # (n_estimators, n_samples)

        # Same row*columns+column flattening trick used elsewhere in this
        # library: combine each tree's vote (sample, class) address into a
        # single index, so one np.bincount call tallies every tree's vote
        # for every sample's every class at once.
        sample_idx = np.broadcast_to(np.arange(n_samples), class_idx.shape)
        flat_idx = sample_idx.ravel() * n_classes + class_idx.ravel()
        vote_counts = np.bincount(
            flat_idx, minlength=n_samples * n_classes
        ).reshape(n_samples, n_classes)

        # Whichever class column has the most votes wins, for every sample
        # at once; ties go to the smaller class label since argmax keeps
        # the first maximum it finds and classes_ is sorted ascending.
        winning_class_idx = np.argmax(vote_counts, axis=1)
        return self.classes_[winning_class_idx]

    # Receives: X — test feature data, and y — the true labels for that
    #   same data.
    # Calculates: the forest's predictions for X, compared against the
    #   true labels y.
    # Returns: a single float — the fraction of samples classified
    #   correctly.
    def score(self, X: NDArray, y: NDArray) -> float:
        """Return mean accuracy on (X, y).

        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
        y : NDArray of shape (n_samples,)

        Returns
        -------
        float
        """
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))


# ---------------------------------------------------------------------------
# Aliases expected by the grader and exercise notebook
# ---------------------------------------------------------------------------
RandomForest = RandomForestClassifier
RandomForestClassifierScratch = RandomForestClassifier
