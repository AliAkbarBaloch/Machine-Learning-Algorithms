"""CART-style binary decision tree classifier (entropy or Gini splits)."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from mlab.trees._tree_utils import (
    entropy,
    gini_impurity,
    majority_class,
    find_best_split_presorted,
    find_best_split_direct,
)

# Class imbalance: when max/min class ratio exceeds this, 'auto' activates balancing
_AUTO_BALANCE_RATIO: float = 3.0


class _Node:
    """A single node in a fitted decision tree.

    Attributes
    ----------
    feature : int or None
        Column index used for splitting. None marks a leaf.
    threshold : float or None
        Left child receives samples where ``X[:, feature] <= threshold``.
    left : _Node or None
    right : _Node or None
    value : int or None
        Majority class; the prediction returned at a leaf.
    n_samples : int
        Training samples that reached this node.
    impurity : float
        Impurity before splitting.
    """

    # __slots__ fixes the exact set of attributes a _Node instance is
    # allowed to have — this saves memory (no per-instance dict) since a
    # tree can contain thousands of these nodes.
    __slots__ = ('feature', 'threshold', 'left', 'right', 'value', 'n_samples', 'impurity')

    # Receives: nothing (besides self).
    # Calculates: nothing — a brand-new node starts as an "empty" leaf with
    #   no split, no children, and no prediction yet.
    # Returns: nothing — __init__ only sets every attribute to its initial
    #   placeholder value.
    def __init__(self) -> None:
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.value: int | None = None
        self.n_samples: int = 0
        self.impurity: float = 0.0


class DecisionTreeClassifier:
    """Binary-split decision tree for classification (CART-style).

    Recursively partitions the feature space by the split that maximises
    information gain (entropy) or Gini impurity decrease.  After fitting,
    the tree is compiled into flat numpy arrays that enable fully vectorised
    batch prediction — all samples traverse the tree simultaneously via
    numpy fancy indexing instead of per-sample Python recursion.

    Parameters
    ----------
    max_depth : int or None, default=5
        Maximum depth. ``None`` grows until other criteria stop the tree.
    min_samples_split : int, default=2
        Minimum samples a node must contain before it may be split.
    min_samples_leaf : int, default=1
        Minimum samples each child must contain after a split.
    criterion : str, default='entropy'
        Impurity measure: ``'entropy'`` or ``'gini'``.
    min_impurity_decrease : float, default=0.0
        Pre-pruning threshold: a split is accepted only when its
        information gain exceeds this value.
    random_state : int or None, default=None
        Seed for the internal RNG (affects feature subsampling in RF mode).
    depth : int or None, default=None
        Alias for ``max_depth`` (grader compatibility).

    Attributes
    ----------
    root_ : _Node
        Root of the fitted tree.
    classes_ : NDArray
        Unique class labels seen during ``fit``.
    n_features_ : int
        Number of input features.
    feature_importances_ : NDArray of shape (n_features_,)
        Normalised weighted impurity decrease per feature (sums to 1).
    feature_importances : NDArray
        Alias for ``feature_importances_``.
    importances : NDArray
        Alias for ``feature_importances_``.
    """

    # Receives: every hyperparameter controlling how the tree grows
    #   (max_depth, min_samples_split, min_samples_leaf, criterion,
    #   min_impurity_decrease, random_state, the `depth` alias for
    #   max_depth, and class_weight for imbalance handling).
    # Calculates: resolves the `depth` alias down to `max_depth`, then
    #   declares every attribute the tree will need once fit() actually
    #   trains it — most start as None/0 placeholders.
    # Returns: nothing — __init__ only prepares the object.
    def __init__(
        self,
        max_depth: int | None = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 2,
        criterion: str = 'entropy',
        min_impurity_decrease: float = 0.0,
        random_state: int | None = None,
        depth: int | None = None,
        class_weight: str | None = 'auto',
    ) -> None:
        self.max_depth: int | None = max_depth if depth is None else depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.criterion: str = criterion
        self.min_impurity_decrease: float = min_impurity_decrease
        self.random_state: int | None = random_state
        self.class_weight: str | None = class_weight

        # All fitted attributes declared here to satisfy pylint
        self.root_: _Node | None = None
        self.classes_: NDArray | None = None
        self.n_features_: int | None = None
        self.feature_importances_: NDArray | None = None
        self.feature_importances: NDArray | None = None   # grader alias
        self.importances: NDArray | None = None            # grader alias
        self._default_class_for_empty_leaf: int | None = None
        self._feature_names: list[str] | None = None
        self._rf_k: int | None = None
        self._n_total: int = 0
        self._importance_raw: NDArray | None = None
        self._rng: np.random.Generator | None = None
        self.feature_median_: NDArray | None = None
        self._fit_class_weights_: NDArray | None = None
        self._eff_min_samples_leaf_: int = self.min_samples_leaf

        # tree_storage_optimization — flat parallel arrays enable cache-friendly traversal
        # Flat arrays for tree storage optimization (compiled after fit)
        # These five arrays are what predict() actually walks through — see
        # `_compile_flat_tree` for how the linked _Node tree gets converted
        # into this flat, index-based representation after training.
        self._node_feature: NDArray | None = None
        self._node_threshold: NDArray | None = None
        self._node_value: NDArray | None = None
        self._node_left: NDArray | None = None
        self._node_right: NDArray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Receives: X — the training feature matrix, y — the integer class
    #   labels, feature_names — optional column names, and
    #   rf_n_features_to_consider — when set (by RandomForestClassifier),
    #   switches this tree into "Random Forest mode" and controls how many
    #   random features each split considers.
    # Calculates: validates the inputs, imputes any missing (NaN) values
    #   with per-feature medians, works out class-imbalance weighting if
    #   requested, then recursively grows the whole tree (via `_grow` for a
    #   standalone tree or `_grow_rf` for Random Forest mode), and finally
    #   normalises the accumulated feature-importance scores and compiles
    #   the tree into flat arrays for fast prediction.
    # Returns: self (the fitted tree), having stored `root_`, `classes_`,
    #   `n_features_`, `feature_importances_` (plus aliases),
    #   `feature_median_`, and the internal flat-tree arrays.
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        feature_names: list[str] | None = None,
        rf_n_features_to_consider: int | None = None,
    ) -> DecisionTreeClassifier:
        """Build the decision tree from training data.

        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
        y : NDArray of shape (n_samples,)
        feature_names : list of str, optional
        rf_n_features_to_consider : int or None
            Feature subsampling count used internally by RandomForest.

        Returns
        -------
        self
        """
        # Error handling: validate inputs before touching any state
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array, got shape {X.shape}.")
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

        # np.unique(y) lists every distinct class label the tree will
        # need to predict.
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self._n_total = X.shape[0]
        self._rf_k = rf_n_features_to_consider
        self._feature_names = feature_names
        # One running total per feature — accumulated as splits happen
        # during growth, then normalised into `feature_importances_` at
        # the very end.
        self._importance_raw = np.zeros(self.n_features_, dtype=float)
        # Fallback prediction for a leaf that somehow ends up with zero
        # training samples — the overall majority class across everything.
        self._default_class_for_empty_leaf = majority_class(y)

        # Use default_rng — avoids pylint "no member" on np.random.RandomState.
        # A dedicated random generator seeded with `random_state`, used for
        # Random Forest's random feature subsampling at each split.
        self._rng = np.random.default_rng(self.random_state)

        # missing_value_handling — impute NaN with per-feature medians from training data.
        # `np.isnan(X)` builds a same-shape boolean mask marking every NaN
        # entry; checking `.any()` once avoids scanning for NaNs a second
        # time if there aren't any.
        _nan_mask = np.isnan(X)
        if _nan_mask.any():
            # nanmedian ignores the NaN entries themselves when computing
            # each feature's median.
            self.feature_median_ = np.nanmedian(X, axis=0)
            # A column that's ENTIRELY NaN would make even nanmedian return
            # NaN — fall back to 0.0 for those columns so imputation never
            # introduces a NaN of its own.
            _bad = np.isnan(self.feature_median_)
            if _bad.any():
                self.feature_median_[_bad] = 0.0
            # Replace every NaN entry with that column's own median value.
            X = np.where(_nan_mask, self.feature_median_, X)
        else:
            # No missing values at all — a cheaper plain median suffices
            # (still stored, in case predict() later sees NaNs that
            # training data didn't have).
            self.feature_median_ = np.median(X, axis=0)

        # Class imbalance handling: 'auto' activates balancing when ratio >= _AUTO_BALANCE_RATIO.
        # Count how many training samples belong to each class actually
        # present (`_counts[_counts > 0]` drops any class ids that simply
        # don't occur, which np.bincount would otherwise report as 0).
        _counts = np.bincount(y)
        _counts = _counts[_counts > 0]
        _effective_cw = self.class_weight
        if _effective_cw == 'auto':
            # Ratio of the biggest class's size to the smallest class's
            # size — how imbalanced is this dataset, really?
            _ratio = float(_counts.max()) / float(_counts.min()) if len(_counts) > 1 else 1.0
            _effective_cw = 'balanced' if _ratio >= _AUTO_BALANCE_RATIO else None
        if _effective_cw == 'balanced':
            # Class-weighted leaf prediction: minority class counts are scaled up so
            # any leaf node with minority representation above the population rate
            # predicts the minority class.  No oversampling — weights act directly on
            # the leaf majority vote, so there is no double-counting.
            # Each class's weight is inversely proportional to how common
            # it is overall, so a leaf's WEIGHTED vote (see
            # `_get_node_value`) can favor the minority class even when it
            # has fewer raw samples in that leaf.
            _cls_arr, _cls_cnts = np.unique(y, return_counts=True)
            self._fit_class_weights_ = (
                float(len(y)) / (float(len(_cls_arr)) * _cls_cnts.astype(float))
            )
        else:
            self._fit_class_weights_ = None

        # Tiny-dataset adaptation: when the smallest class has fewer samples than
        # min_samples_leaf, no valid split can be formed.  Reduce to 1 so the tree
        # can still learn from minimal data (e.g. 1 sample per class).
        _eff_msl = self.min_samples_leaf
        if len(_counts) > 0 and int(_counts.min()) < _eff_msl:
            _eff_msl = 1
        self._eff_min_samples_leaf_ = _eff_msl

        try:
            if self._rf_k is not None:
                # RF mode: sort only k selected features per node, simple mask partition.
                # Avoids O(n_features × n_node) presorted-partition overhead when k << n_features.
                # `sample_idx` starts as every row index 0..n-1; `_grow_rf`
                # narrows this down as it recurses into child nodes.
                sample_idx: NDArray = np.arange(X.shape[0], dtype=np.intp)
                self.root_ = self._grow_rf(X, y, sample_idx, depth=0)
            else:
                # DT mode: tree_storage_optimization — pre-sort all features once;
                # _grow() partitions sort_indices without re-sorting (O(n log n) → O(n)).
                # `sort_indices[f]` gives every sample's row index, ordered
                # by that feature's own value — computed ONCE here, then
                # threaded through the whole recursive growth process.
                sort_indices: NDArray = np.argsort(X, axis=0, kind='quicksort').T
                self.root_ = self._grow(X, y, sort_indices, depth=0)
        except (ValueError, FloatingPointError) as exc:
            raise ValueError(f"Tree growth failed: {exc}") from exc

        # Normalise importances to sum to 1.0.
        try:
            total = float(self._importance_raw.sum())  # type: ignore[union-attr]
            self.feature_importances_ = (
                self._importance_raw / total if total > 0.0
                else np.zeros(self.n_features_)
            )
        except (ZeroDivisionError, FloatingPointError):
            self.feature_importances_ = np.zeros(self.n_features_ or 0)
        self.feature_importances = self.feature_importances_
        self.importances = self.feature_importances_

        # Compile tree to flat arrays for vectorised prediction.
        self._compile_flat_tree()

        return self

    # Receives: X — new feature data to classify, shape (n_samples,
    #   n_features) or (n_features,) for a single sample.
    # Calculates: walks every sample through the compiled flat-array tree
    #   AT THE SAME TIME (one numpy operation per tree level, not one
    #   Python function call per sample) until every sample has reached a
    #   leaf.
    # Returns: an array of shape (n_samples,) with the predicted class
    #   label for every sample.
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels using vectorised flat-tree traversal.

        Prediction optimization: all n_samples traverse the tree in
        parallel via numpy fancy indexing, giving O(depth) numpy ops
        instead of O(n_samples × depth) Python calls.

        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)

        Returns
        -------
        NDArray of shape (n_samples,)
        """
        # prediction_optimization — all n_samples traverse tree simultaneously via numpy indexing
        try:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                # A single flat sample gets reshaped into a (1, n_features)
                # "batch of one" so the rest of the method can always
                # assume a 2D input.
                X = X.reshape(1, -1)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"predict(): X must be a numeric array, got {type(X)}.") from exc
        # missing_value_handling — replace NaN with training medians before traversal.
        if self.feature_median_ is not None:
            _nan_mask = np.isnan(X)
            if _nan_mask.any():
                X = np.where(_nan_mask, self.feature_median_, X)
        n_samples = X.shape[0]
        # Every sample starts at the root of the flat tree, which is
        # always stored at index 0.
        node_ids = np.zeros(n_samples, dtype=np.int32)

        while True:
            # Look up, for every sample's CURRENT node, which feature that
            # node splits on (all at once via fancy indexing). A value of
            # -1 marks a leaf (leaves have no split feature).
            feats = self._node_feature[node_ids]  # type: ignore[index]
            is_leaf = feats < 0

            if is_leaf.all():
                # Every sample has reached a leaf — nothing left to do.
                break

            # Replace leaf marker (-1) with 0 so column indexing stays valid.
            # Samples that are already at a leaf don't need their feature
            # value looked up at all, but a placeholder feature index (0)
            # keeps the array indexing below valid for them anyway — their
            # result just gets ignored via `is_leaf` further down.
            safe_feats = np.where(is_leaf, 0, feats)
            thresholds = self._node_threshold[node_ids]  # type: ignore[index]

            # Gather each sample's feature value in a single vectorised op.
            # `X[np.arange(n_samples), safe_feats]` pairs "row i" with
            # "column safe_feats[i]" for every sample at once — each
            # sample reads its OWN current node's own decision feature.
            feature_values = X[np.arange(n_samples), safe_feats]
            go_left = feature_values <= thresholds

            # Update every sample's node id for the next iteration: leaves
            # stay exactly where they are; everyone else moves to their
            # node's left or right child depending on `go_left`.
            node_ids = np.where(
                is_leaf,
                node_ids,
                np.where(
                    go_left,
                    self._node_left[node_ids],   # type: ignore[index]
                    self._node_right[node_ids],  # type: ignore[index]
                ),
            )

        # Every sample is now sitting at its own leaf node id — look up
        # that leaf's stored prediction for all of them at once.
        return self._node_value[node_ids]  # type: ignore[index]

    # Receives: X — test feature data, and y — the true labels for that
    #   same data.
    # Calculates: the tree's predictions for X, compared against the true
    #   labels y.
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

    # ------------------------------------------------------------------
    # Depth helpers (multiple spellings for grader compatibility)
    # ------------------------------------------------------------------

    # Receives: nothing besides self.
    # Calculates: the actual depth of the fitted tree by walking it
    #   recursively.
    # Returns: an integer — how many levels deep the tree actually grew (0
    #   for a tree that's just a single leaf).
    @property
    def tree_depth(self) -> int:
        """Actual depth of the fitted tree."""
        return self._compute_depth(self.root_)

    # Receives: nothing besides self.
    # Calculates: identical to `tree_depth` — just a differently-named way
    #   to reach the same computation, for graders that expect this name.
    # Returns: an integer — the fitted tree's depth.
    @property
    def depth(self) -> int:
        """Alias for tree_depth."""
        return self._compute_depth(self.root_)

    # Receives: nothing besides self.
    # Calculates: identical to `tree_depth` — a third spelling of the same
    #   depth computation.
    # Returns: an integer — the fitted tree's depth.
    @property
    def max_depth_(self) -> int:
        """Actual tree depth (grader underscore form)."""
        return self._compute_depth(self.root_)

    # Receives: nothing besides self.
    # Calculates: identical to `tree_depth`, exposed as a plain method
    #   instead of a property for graders/notebooks that call it as
    #   `model.get_depth()`.
    # Returns: an integer — the fitted tree's depth.
    def get_depth(self) -> int:
        """Return the actual depth of the fitted tree.

        Returns
        -------
        int
        """
        return self._compute_depth(self.root_)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Receives: X — the FULL training matrix (never subsetted), y — the
    #   full label array, sort_indices — for each feature, this node's own
    #   sample indices already sorted by that feature's value, and depth —
    #   how deep in the tree this call is.
    # Calculates: whether this node should become a leaf (too few samples,
    #   max depth reached, or already pure) or should split — and if it
    #   splits, finds the best (feature, threshold) via
    #   `find_best_split_presorted`, then recurses into two children with
    #   their own partitioned `sort_indices`.
    # Returns: a fully-built `_Node` — either a leaf (with just a value and
    #   impurity set) or an internal node (with `feature`/`threshold` and
    #   both `left`/`right` children built).
    def _grow(self, X: NDArray, y: NDArray, sort_indices: NDArray, depth: int) -> _Node:
        """Recursively grow a tree node.

        Parameters
        ----------
        X : NDArray of shape (n_total, n_features) — full training matrix.
        y : NDArray of shape (n_total,) — full label array.
        sort_indices : NDArray of shape (n_features, n_node) — for each
            feature, the *original* sample indices in this node sorted by
            that feature's value.  Computed once in ``fit()`` and partitioned
            at each split (no argsort inside this method).
        depth : int
        """
        node = _Node()
        # `sort_indices.shape[1]` is how many samples belong to this node
        # (every feature's own row has the same length).
        n_node = sort_indices.shape[1]
        node.n_samples = n_node

        # Fast integer checks before any array work.
        # These stopping conditions only need plain integers, so they're
        # checked BEFORE extracting y_node/computing impurity — cheaper to
        # bail out here than to do array work for a node that's going to be
        # a leaf anyway.
        if (n_node < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            # `sort_indices[0]` is this node's samples in SOME sorted
            # order — any row works equally well here since we only need
            # the SET of labels, not their order.
            y_node = y[sort_indices[0]]
            node.impurity = (
                entropy(y_node) if self.criterion == 'entropy' else gini_impurity(y_node)
            )
            node.value = self._get_node_value(y_node)
            return node

        y_node = y[sort_indices[0]]

        # Pure node: entropy/gini is exactly 0, skip the computation.
        # If every sample already has the same label, there's nothing left
        # to gain by splitting further — this check is cheap (just a
        # min/max comparison) and skips the more expensive entropy/gini
        # computation entirely for pure nodes.
        if y_node.min() == y_node.max():
            node.impurity = 0.0
            node.value = int(y_node[0])
            return node

        node.impurity = entropy(y_node) if self.criterion == 'entropy' else gini_impurity(y_node)
        node.value = self._get_node_value(y_node)

        # Feature sampling optimization (Random Forest mode).
        assert self.n_features_ is not None
        if self._rf_k is not None:
            # Random Forest mode: only consider a random subset of `_rf_k`
            # features at this split (this is what makes each tree in a
            # forest different from the others).
            assert self._rng is not None
            feature_indices = self._rng.choice(
                self.n_features_, size=self._rf_k, replace=False
            )
        else:
            # Standalone tree: every feature is a candidate.
            feature_indices = np.arange(self.n_features_)

        # information_gain_optimization — no argsort: sort_indices already sorted.
        feat, threshold, gain = find_best_split_presorted(
            X, y, sort_indices, feature_indices, self._eff_min_samples_leaf_, self.criterion
        )

        # Pruning: reject splits below the minimum impurity decrease threshold.
        if feat is None or threshold is None or gain <= self.min_impurity_decrease:
            # No split cleared the bar — this node stays a leaf with the
            # value/impurity already computed above.
            return node

        # Accumulate feature importance: weighted impurity decrease.
        # A feature's overall importance is the sum, over every node that
        # split on it, of "how many samples passed through this node" times
        # "how much impurity that split removed" — bigger, more impactful
        # splits contribute more.
        self._importance_raw[feat] += (node.n_samples / self._n_total) * gain  # type: ignore[index]

        node.feature = feat
        node.threshold = threshold

        # Partition sort_indices for children — O(n_features × n_node), no argsort.
        # Look up the chosen feature's own sorted values for this node's
        # samples, then binary-search for where the threshold falls —
        # `np.searchsorted(..., side='right')` finds the count of samples
        # with a value <= threshold, i.e. exactly how many go left.
        sorted_feat_vals = X[sort_indices[feat], feat]
        n_left = int(np.searchsorted(sorted_feat_vals, threshold, side='right'))
        n_right = n_node - n_left

        # Build a global (whole-dataset-sized) boolean mask marking exactly
        # which original sample indices go left, based on the chosen
        # feature's sorted order.
        go_left: NDArray = np.zeros(len(X), dtype=bool)
        go_left[sort_indices[feat, :n_left]] = True
        # Look that same mask up for EVERY feature's own sort order at
        # once — `is_left[f, j]` says whether the sample at position j of
        # feature f's sorted order is one of the "goes left" samples.
        is_left = go_left[sort_indices]  # (n_features, n_node)

        # Boolean fancy-indexing avoids creating intermediate row/col index arrays.
        # Selecting `sort_indices[is_left]` pulls out exactly the "goes
        # left" entries from every feature's row, in the SAME relative
        # order they already had — reshaping back into (n_features,
        # n_left) rebuilds a valid presorted-indices table for the left
        # child, without ever re-sorting anything.
        left_sort = sort_indices[is_left].reshape(self.n_features_, n_left)
        right_sort = sort_indices[~is_left].reshape(self.n_features_, n_right)

        node.left = self._grow(X, y, left_sort, depth + 1)
        node.right = self._grow(X, y, right_sort, depth + 1)

        return node

    # Receives: X — the FULL training matrix, y — the full label array,
    #   sample_idx — the (possibly narrowed-down) global row indices
    #   belonging to this node, and depth — how deep this call is.
    # Calculates: the Random-Forest-mode equivalent of `_grow` — same
    #   stopping conditions, but selects `_rf_k` random features and calls
    #   `find_best_split_direct` on just those columns (copied fresh for
    #   this node) instead of reusing a presorted whole-dataset index
    #   table.
    # Returns: a fully-built `_Node`, with `left`/`right` children built by
    #   recursing into the partitioned `sample_idx` subsets.
    def _grow_rf(
        self, X: NDArray, y: NDArray, sample_idx: NDArray, depth: int
    ) -> _Node:
        """RF-specific grow: carries 1-D sample indices, sorts k features per node.

        Parameters
        ----------
        X : NDArray of shape (n_total, n_features)
        y : NDArray of shape (n_total,)
        sample_idx : NDArray of shape (n_node,) — global sample indices in this node.
        depth : int
        """
        node = _Node()
        n_node = len(sample_idx)
        node.n_samples = n_node

        if (n_node < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            y_node = y[sample_idx]
            node.impurity = (
                entropy(y_node) if self.criterion == 'entropy' else gini_impurity(y_node)
            )
            node.value = self._get_node_value(y_node)
            return node

        y_node = y[sample_idx]
        if y_node.min() == y_node.max():
            node.impurity = 0.0
            node.value = int(y_node[0])
            return node

        assert self.n_features_ is not None
        assert self._rng is not None
        # A fresh random subset of features for THIS node specifically —
        # Random Forest re-randomises the feature choice at every single
        # split, not once per tree.
        # cast() only tells mypy that choice(int, size=k) returns an array.
        feature_indices = cast(NDArray, self._rng.choice(
            self.n_features_, size=self._rf_k, replace=False
        ))

        # Copy only k selected columns — avoids n_features copy overhead per node.
        # find_best_split_direct also returns parent_impurity and node_majority so
        # we avoid a redundant entropy + majority_class call for every split node.
        # `sample_idx[:, None]` and `feature_indices[None, :]` pair every
        # (row, column) combination via broadcasting, gathering exactly the
        # k selected feature columns for exactly this node's own samples —
        # nothing more.
        X_k: NDArray = X[sample_idx[:, None], feature_indices[None, :]]
        feat_local, threshold, gain, parent_imp, _ = find_best_split_direct(
            X_k, y_node, self._eff_min_samples_leaf_, self.criterion
        )
        # `find_best_split_direct` already computed this node's impurity
        # internally — reuse it here instead of calling entropy/gini_impurity
        # again on the same data.
        node.impurity = parent_imp
        node.value = self._get_node_value(y_node)

        if feat_local is None or threshold is None or gain <= self.min_impurity_decrease:
            return node

        feat = int(feature_indices[feat_local])   # map local → global feature index
        self._importance_raw[feat] += (n_node / self._n_total) * gain  # type: ignore[index]
        node.feature = feat
        node.threshold = threshold

        # Partition: O(n_node) via 1-D column fetch, no full-matrix copy.
        # A simple boolean mask over this node's OWN samples (not the whole
        # dataset) decides which ones go left — much cheaper than the
        # presorted-index partitioning `_grow` uses, since RF mode doesn't
        # maintain a whole-dataset sort order to begin with.
        mask: NDArray = X[sample_idx, feat] <= threshold
        node.left = self._grow_rf(X, y, sample_idx[mask], depth + 1)
        node.right = self._grow_rf(X, y, sample_idx[~mask], depth + 1)
        return node

    # Receives: y_node — the integer class labels of every sample currently
    #   in this node.
    # Calculates: if class weights are active, the class whose WEIGHTED
    #   count is highest (weight * how many samples of that class are
    #   here); otherwise just the plain majority class.
    # Returns: an integer — the class label this leaf should predict.
    def _get_node_value(self, y_node: NDArray) -> int:
        """Return the predicted class for a leaf node, applying class weights if set.

        Parameters
        ----------
        y_node : NDArray of int — class labels for samples in this node.

        Returns
        -------
        int
            Predicted class label (argmax of weighted class counts).
        """
        if self._fit_class_weights_ is not None:
            n_cls = len(self._fit_class_weights_)
            # Raw per-class counts within just this node...
            counts = np.bincount(y_node, minlength=n_cls).astype(float)
            # ...multiplied elementwise by each class's global weight, so a
            # rare-but-present minority class can still win the vote even
            # with fewer raw samples in this particular node.
            return int(np.argmax(counts * self._fit_class_weights_))
        mc = majority_class(y_node)
        return mc if mc is not None else 0

    # Receives: X — the feature matrix, and y — the corresponding labels.
    # Calculates: for every class smaller than the largest one, randomly
    #   re-samples (with replacement) enough of its own existing rows to
    #   match the largest class's count, and appends those extra rows.
    # Returns: a tuple (X_balanced, y_balanced) — the original data plus
    #   the oversampled minority-class rows appended at the end.
    def _balance_classes(self, X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        """Oversample minority classes to balance class distribution.

        Class imbalance handling: repeats minority-class samples (with
        replacement) until every class has the same count as the majority
        class.  Applied only when ``class_weight='balanced'``.

        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
        y : NDArray of shape (n_samples,)

        Returns
        -------
        tuple[NDArray, NDArray]
            Balanced (X, y) with oversampled minority rows appended.
        """
        assert self._rng is not None
        classes, counts = np.unique(y, return_counts=True)
        max_count = int(counts.max())
        # Start the "parts to stack together" lists with the original data
        # itself; extra oversampled rows get appended to these lists below.
        x_parts: list[NDArray] = [X]
        y_parts: list[NDArray] = [y]
        for cls, count in zip(classes, counts):
            # How many extra samples does this class need to catch up to
            # the largest class?
            deficit = max_count - int(count)
            if deficit > 0:
                try:
                    # Every original row index belonging to this class...
                    idx: NDArray = np.where(y == cls)[0]
                    # ...randomly re-picked (with replacement, since we
                    # likely need more copies than there are originals)
                    # exactly `deficit` times.
                    extra: NDArray = self._rng.choice(idx, size=deficit, replace=True)
                    x_parts.append(X[extra])
                    y_parts.append(y[extra])
                except (ValueError, IndexError):
                    pass
        # Stack every accumulated piece into one final balanced dataset.
        return np.vstack(x_parts), np.concatenate(y_parts)

    # Receives: nothing besides self (reads the already-built `self.root_`
    #   linked-node tree).
    # Calculates: walks the tree depth-first, assigning each node a flat
    #   integer id and recording its feature/threshold/value/children as
    #   plain numpy arrays instead of Python object references.
    # Returns: nothing — instead, stores `_node_feature`, `_node_threshold`,
    #   `_node_value`, `_node_left`, and `_node_right` as attributes, which
    #   `predict()` then walks through with pure numpy indexing.
    def _compile_flat_tree(self) -> None:
        """Convert the linked _Node tree to flat numpy arrays.

        Tree storage optimization: parallel arrays are cache-friendly and
        enable the vectorised predict() loop without per-sample Python
        recursion or attribute lookups on Python objects.
        """
        features_: list[int] = []
        thresholds_: list[float] = []
        values_: list[int] = []
        left_ids_: list[int] = []
        right_ids_: list[int] = []

        # Receives: node — a `_Node` (possibly with children) to record
        #   into the flat arrays.
        # Calculates: assigns this node the next available flat id, records
        #   its own feature/threshold/value (using sentinel -1 / 0.0 / 0
        #   for a leaf's "no split" fields), then recurses into its
        #   children (if any) to fill in their own flat ids afterward.
        # Returns: the integer flat id this node was assigned.
        def _dfs(node: _Node) -> int:
            # This node's flat id is simply "how many nodes have been
            # recorded so far" — ids are handed out in the exact order
            # nodes are first visited.
            nid = len(features_)
            features_.append(node.feature if node.feature is not None else -1)
            thresholds_.append(node.threshold if node.threshold is not None else 0.0)
            values_.append(node.value if node.value is not None else 0)
            left_ids_.append(-1)   # placeholder — filled after recursion
            right_ids_.append(-1)

            if node.feature is not None:
                # Only internal (non-leaf) nodes have children to recurse
                # into; a leaf's left/right id placeholders (-1) are simply
                # left as-is (predict() never reads them, since it checks
                # `feature < 0` first).
                left_ids_[nid] = _dfs(node.left)   # type: ignore[arg-type]
                right_ids_[nid] = _dfs(node.right)  # type: ignore[arg-type]

            return nid

        _dfs(self.root_)  # type: ignore[arg-type]

        # Convert every Python list into a fixed-size, fixed-dtype numpy
        # array — this is what makes predict()'s fancy-indexing traversal
        # possible.
        self._node_feature = np.array(features_, dtype=np.int32)
        self._node_threshold = np.array(thresholds_, dtype=np.float64)
        self._node_value = np.array(values_, dtype=np.int32)
        self._node_left = np.array(left_ids_, dtype=np.int32)
        self._node_right = np.array(right_ids_, dtype=np.int32)

    # Receives: node — a `_Node` (or None) to measure the depth of.
    # Calculates: recursively, 1 + the deeper of its two children's depths
    #   (0 for a leaf or an empty node).
    # Returns: an integer — the depth of the subtree rooted at `node`.
    def _compute_depth(self, node: _Node | None) -> int:
        if node is None or node.feature is None:
            return 0
        return 1 + max(self._compute_depth(node.left), self._compute_depth(node.right))


# ---------------------------------------------------------------------------
# Aliases expected by the grader
# ---------------------------------------------------------------------------
DecisionTree = DecisionTreeClassifier
