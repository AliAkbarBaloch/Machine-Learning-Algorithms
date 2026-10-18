"""Impurity measures and vectorised best-split search shared by the tree models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Numerical stability: small constant guards log2(0) in entropy calculations
_EPS: float = 1e-15


# ---------------------------------------------------------------------------
# Scalar impurity helpers (used for single nodes in _grow)
# ---------------------------------------------------------------------------

# Receives: counts — how many samples of each class are in this node
#   (one integer per class), and n — the total number of samples in the
#   node (= sum of counts).
# Calculates: the Shannon entropy of the class distribution, using only the
#   classes that actually have at least one sample (to avoid log2(0)).
# Returns: a single float — 0.0 for an empty or perfectly pure node
#   (all one class), higher values the more mixed the classes are.
def _entropy_from_counts(counts: NDArray, n: int) -> float:
    """Shannon entropy from raw class counts.

    Numerical stability: zero-count classes are excluded before the
    logarithm, preventing log2(0) = -inf.

    Parameters
    ----------
    counts : NDArray of int
    n : int

    Returns
    -------
    float
    """
    if n == 0:
        return 0.0
    # `counts > 0` keeps only classes that actually appear in this node
    # (dividing a zero-count class by n would give probability 0, and
    # log2(0) is -infinity — excluding it avoids that entirely, since a
    # class with 0 samples contributes 0 to the entropy sum anyway).
    probs = counts[counts > 0] / n
    # Shannon entropy: -sum(p * log2(p)) over every class's probability.
    # `_EPS` is added purely as an extra numerical-stability cushion (the
    # `counts > 0` filter already prevents an exact log2(0), but very tiny
    # probabilities could still make log2 return large negative numbers).
    return float(-np.sum(probs * np.log2(probs + _EPS)))


# Receives: counts — per-class sample counts in this node, and n — the
#   total number of samples in the node.
# Calculates: the Gini impurity of the class distribution — the
#   probability that two randomly picked samples from this node would
#   belong to different classes.
# Returns: a single float — 0.0 for an empty or perfectly pure node, up to
#   0.5 for a maximally mixed binary split.
def _gini_from_counts(counts: NDArray, n: int) -> float:
    """Gini impurity from raw class counts.

    Parameters
    ----------
    counts : NDArray of int
    n : int

    Returns
    -------
    float
    """
    if n == 0:
        return 0.0
    # Convert raw counts into per-class probabilities.
    probs = counts / n
    # Gini = 1 - sum(p_i^2). `np.dot(probs, probs)` computes sum(p_i * p_i)
    # in one call — mathematically identical to `np.sum(probs ** 2)`, just
    # using the dot-product primitive instead.
    return float(1.0 - np.dot(probs, probs))


# ---------------------------------------------------------------------------
# Vectorised impurity helpers (used for batch split search)
# ---------------------------------------------------------------------------

# Receives: counts_2d — a table of per-class counts, one row per candidate
#   split (m splits total), one column per class; and ns — the total
#   sample count for each of those m splits.
# Calculates: the Shannon entropy for every one of the m splits at once,
#   using numpy broadcasting instead of looping over splits in Python.
# Returns: an array of shape (m,) — one entropy value per candidate split.
def _batch_entropy(counts_2d: NDArray, ns: NDArray) -> NDArray:
    """Vectorised Shannon entropy for m independent partitions.

    Information gain optimization: computes entropy for all candidate
    splits simultaneously using numpy broadcasting rather than a Python
    loop — O(m * n_classes) numpy operations for all m splits at once.

    Parameters
    ----------
    counts_2d : NDArray of shape (m, n_classes)
    ns : NDArray of shape (m,)

    Returns
    -------
    NDArray of shape (m,)
    """
    # Numerical stability: clamp denominator to ≥1 to avoid divide-by-zero.
    # `ns[:, None]` reshapes ns from (m,) to (m, 1) so it broadcasts against
    # counts_2d's (m, n_classes) shape — every row's own total divides that
    # row's own counts, all in one operation.
    probs = counts_2d / np.maximum(ns[:, None], 1)
    # np.where(condition, value_if_true, value_if_false): wherever a
    # probability is actually > 0, compute its log2 term as normal; wherever
    # it's 0 (a class absent from that split), use 0 instead — since a
    # class with probability 0 contributes nothing to entropy, and log2(0)
    # would otherwise be -infinity.
    log_probs = np.where(probs > 0, np.log2(np.maximum(probs, _EPS)), 0.0)
    # Sum over axis=1 (across classes, within each split/row) gives one
    # entropy value per split.
    return -np.sum(probs * log_probs, axis=1)


# Receives: counts_2d — per-class counts for m candidate splits, and ns —
#   the total sample count for each split.
# Calculates: the Gini impurity for every one of the m splits at once.
# Returns: an array of shape (m,) — one Gini impurity value per split.
def _batch_gini(counts_2d: NDArray, ns: NDArray) -> NDArray:
    """Vectorised Gini impurity for m independent partitions.

    Parameters
    ----------
    counts_2d : NDArray of shape (m, n_classes)
    ns : NDArray of shape (m,)

    Returns
    -------
    NDArray of shape (m,)
    """
    # Same row-wise broadcast division as _batch_entropy: each split's own
    # counts divided by that split's own total.
    probs = counts_2d / np.maximum(ns[:, None], 1)
    # Gini = 1 - sum(p_i^2) per row; summing `probs * probs` over axis=1
    # (across classes) gives that sum for every split at once.
    return 1.0 - np.sum(probs * probs, axis=1)


# Receives: parent_imp (the entropy of the node before splitting), n (total
#   samples in the node), n_lefts/n_rights (how many samples would land in
#   each child for every one of m candidate splits), and lp/ln/rp/rn — the
#   positive-class and negative-class counts on each side of each of those
#   m candidate splits.
# Calculates: the information gain (reduction in entropy) for every one of
#   the m candidate splits at once, using the fact that with only 2 classes
#   each child's entropy needs just two numbers (not a full class-count
#   vector).
# Returns: an array of shape (m,) — the information gain for each candidate
#   split.
def _binary_entropy_gain(
    parent_imp: float,
    n: int,
    n_lefts: NDArray,
    n_rights: NDArray,
    lp: NDArray,
    ln: NDArray,
    rp: NDArray,
    rn: NDArray,
) -> NDArray:
    """Vectorised information gain for binary-label splits.

    Avoids the n_classes-dimension entirely: each child's entropy is
    computed from two scalars (positive-count, negative-count) rather
    than a (m, 2) matrix, halving memory traffic vs _batch_entropy.

    Parameters
    ----------
    parent_imp : float
    n : int — total node samples
    n_lefts, n_rights : NDArray of shape (m,)
    lp, ln : NDArray — left positive / negative counts
    rp, rn : NDArray — right positive / negative counts

    Returns
    -------
    NDArray of shape (m,) — information gain for each candidate split
    """
    # Numerical stability: clamp denominators.
    # Never divide by an actual 0 — if a child ended up empty (shouldn't
    # happen given min_samples_leaf, but guarded anyway), treat its size as
    # 1 instead so the division below stays finite.
    nl = np.maximum(n_lefts, 1.0)
    nr = np.maximum(n_rights, 1.0)
    # Left entropy.
    # `pl`/`ql` are the left child's positive/negative class probabilities
    # for every candidate split at once.
    pl = lp / nl
    ql = ln / nl
    # Binary entropy = -(p*log2(p) + q*log2(q)); np.where again avoids
    # log2(0) wherever a class is entirely absent from the left child.
    h_l = -(np.where(pl > 0, pl * np.log2(np.maximum(pl, _EPS)), 0.0)
            + np.where(ql > 0, ql * np.log2(np.maximum(ql, _EPS)), 0.0))
    # Right entropy — identical formula, applied to the right child's own
    # counts.
    pr = rp / nr
    qr = rn / nr
    h_r = -(np.where(pr > 0, pr * np.log2(np.maximum(pr, _EPS)), 0.0)
            + np.where(qr > 0, qr * np.log2(np.maximum(qr, _EPS)), 0.0))
    # Information gain = parent's impurity minus the sample-weighted average
    # of the two children's impurities — the standard information-gain
    # formula, computed for every candidate split simultaneously.
    return parent_imp - (n_lefts / n * h_l + n_rights / n * h_r)


# Receives: the same inputs as `_binary_entropy_gain` — parent_imp, n,
#   n_lefts/n_rights, and the binary class counts lp/ln/rp/rn for m
#   candidate splits.
# Calculates: the Gini-impurity gain for every one of the m candidate
#   splits, using the simplified binary formula Gini = 2*p*(1-p).
# Returns: an array of shape (m,) — the Gini gain for each candidate split.
def _binary_gini_gain(
    parent_imp: float,
    n: int,
    n_lefts: NDArray,
    n_rights: NDArray,
    lp: NDArray,
    ln: NDArray,
    rp: NDArray,
    rn: NDArray,
) -> NDArray:
    """Vectorised Gini gain for binary-label splits.

    For binary labels Gini = 2*p*(1-p), so the child impurity reduces
    to a single multiply-add instead of a matrix sum.

    Parameters
    ----------
    parent_imp : float
    n : int
    n_lefts, n_rights, lp, ln, rp, rn : NDArray of shape (m,)

    Returns
    -------
    NDArray of shape (m,)
    """
    nl = np.maximum(n_lefts, 1.0)
    nr = np.maximum(n_rights, 1.0)
    # Left/right child class probabilities, same as in the entropy version.
    pl = lp / nl
    ql = ln / nl
    pr = rp / nr
    qr = rn / nr
    # For exactly two classes, Gini = 1 - (p^2 + q^2) simplifies to
    # 2*p*q (since q = 1-p, this is algebraically the same value, just
    # cheaper to compute as one multiply instead of two squares and a
    # subtraction). The overall gain formula is otherwise identical to the
    # entropy version above.
    return parent_imp - (n_lefts / n * (2.0 * pl * ql) + n_rights / n * (2.0 * pr * qr))


# ---------------------------------------------------------------------------
# Public utility functions
# ---------------------------------------------------------------------------

# Receives: y — an array of integer class labels for a set of samples.
# Calculates: how many samples belong to each class (via np.bincount),
#   then the Shannon entropy of that class distribution.
# Returns: a single float — the entropy in bits (0.0 for an empty or pure
#   set of labels).
def entropy(y: NDArray) -> float:
    """Compute the Shannon entropy of a label array.

    Parameters
    ----------
    y : NDArray of shape (n_samples,)
        Integer class labels.

    Returns
    -------
    float
        Entropy in bits. Zero for a pure or empty node.
    """
    if len(y) == 0:
        return 0.0
    # np.bincount counts how many times each integer 0, 1, 2, ... appears
    # in y — exactly the per-class sample counts entropy needs.
    counts = np.bincount(y.astype(int))
    return _entropy_from_counts(counts, len(y))


# Receives: y — an array of integer class labels for a set of samples.
# Calculates: how many samples belong to each class, then the Gini
#   impurity of that class distribution.
# Returns: a single float — the Gini impurity (0.0 for an empty or pure
#   set of labels, up to 0.5 for a maximally mixed binary set).
def gini_impurity(y: NDArray) -> float:
    """Compute the Gini impurity of a label array.

    Parameters
    ----------
    y : NDArray of shape (n_samples,)
        Integer class labels.

    Returns
    -------
    float
        Gini impurity in [0, 0.5] for binary classification.
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y.astype(int))
    return _gini_from_counts(counts, len(y))


# Receives: y — an array of integer class labels for a set of samples.
# Calculates: how many samples belong to each class, then which class has
#   the most.
# Returns: the integer label of the most frequent class, or None if y is
#   empty.
def majority_class(y: NDArray) -> int | None:
    """Return the most frequent class label.

    Parameters
    ----------
    y : NDArray of shape (n_samples,)
        Integer class labels.

    Returns
    -------
    int or None
        Most frequent label, or None if y is empty.
    """
    if len(y) == 0:
        return None
    # np.bincount(y) gives per-class counts; np.argmax finds which class
    # index has the highest count — i.e. the majority class.
    return int(np.argmax(np.bincount(y.astype(int))))


# Receives: X — the full feature matrix for this node's samples, y — their
#   integer class labels, feature_indices — which columns of X to consider
#   splitting on, min_samples_leaf — the minimum allowed samples per child,
#   and criterion — 'entropy' or 'gini'.
# Calculates: for every candidate feature and every possible threshold
#   within it, the information gain of splitting the node there — all
#   candidate thresholds within a feature are evaluated together via a
#   single cumulative-sum pass instead of a per-sample Python loop.
# Returns: a 3-tuple (best_feature, best_threshold, best_gain) — the single
#   best split found across every feature and threshold considered.
def find_best_split(
    X: NDArray,
    y: NDArray,
    feature_indices: NDArray,
    min_samples_leaf: int,
    criterion: str,
) -> tuple[int | None, float | None, float]:
    """Find the binary split that maximises information gain.

    Information gain optimization — fully vectorised split search:
    instead of a Python loop that updates class counts one sample at a
    time, we compute cumulative class counts for every candidate threshold
    at once using ``np.cumsum`` over a one-hot label matrix.  This reduces
    per-feature work from O(n) Python iterations to O(n * n_classes)
    numpy operations, giving a 10–50× speedup on real datasets.

    Caching strategy: the one-hot matrix ``y_onehot`` is built once per
    node call and reused for every feature — its column-reordering per
    feature is a single fancy-index, not a recomputation.

    Parameters
    ----------
    X : NDArray of shape (n_samples, n_features)
    y : NDArray of shape (n_samples,)
        Integer class labels.
    feature_indices : NDArray of int
        Column indices to consider for splitting.
    min_samples_leaf : int
        Minimum samples that must end up in each child.
    criterion : str
        'entropy' or 'gini'.

    Returns
    -------
    best_feature : int or None
    best_threshold : float or None
    best_gain : float
    """
    n = len(y)
    if n < 2:
        # Can't split a node with fewer than 2 samples — nothing to divide.
        return None, None, 0.0

    # How many distinct classes are present (labels are assumed to be
    # 0..n_classes-1), and how many samples of each class are in this node.
    n_classes = int(y.max()) + 1
    all_counts = np.bincount(y, minlength=n_classes).astype(np.float64)

    # Pick which impurity formula to use, once, based on the criterion
    # string — avoids an if/else check inside the per-feature loop below.
    _scalar_imp = _entropy_from_counts if criterion == 'entropy' else _gini_from_counts
    _batch_imp = _batch_entropy if criterion == 'entropy' else _batch_gini

    # This node's own impurity BEFORE any split — the baseline every
    # candidate split's gain is measured against.
    parent_impurity = _scalar_imp(all_counts, n)

    # information_gain_optimization — cumsum over one-hot matrix eliminates Python loop.
    # A "one-hot" encoding turns each integer label into a row with a single
    # 1 in the column matching that label and 0s elsewhere — e.g. label 2
    # out of 3 classes becomes [0, 0, 1]. Building this once and reusing it
    # for every feature avoids recomputing it n_features times.
    y_onehot: NDArray = np.zeros((n, n_classes), dtype=np.float64)
    y_onehot[np.arange(n), y] = 1.0

    best_feat: int | None = None
    best_thresh: float | None = None
    best_gain: float = 0.0

    for feat in feature_indices:
        col = X[:, feat]

        # Skip constant features — no information gain possible.
        if col.min() == col.max():
            continue

        # Sort by feature value; reuse sort order for label one-hot (caching).
        # `kind='stable'` keeps equal values in their original relative
        # order, which matters for reproducibly picking a threshold among
        # ties later.
        sort_idx = np.argsort(col, kind='stable')
        sorted_col = col[sort_idx]
        # Re-order the SAME one-hot matrix built above using this feature's
        # own sort order, instead of rebuilding it from scratch.
        sorted_y_oh = y_onehot[sort_idx]  # (n, n_classes)

        # Cumulative left class counts: cumsum[i] = counts for samples 0..i.
        # np.cumsum running down each column gives, at row i, "how many of
        # each class have we seen among the first i+1 sorted samples" —
        # exactly the class counts a left child would have if we split
        # right after position i.
        cumsum = np.cumsum(sorted_y_oh, axis=0)  # (n, n_classes)

        # Candidate split positions: indices where the feature value changes.
        # There's no point placing a threshold between two samples with the
        # SAME feature value (they'd always end up on the same side), so
        # only boundaries where consecutive sorted values differ are
        # considered.
        bi = np.where(sorted_col[:-1] != sorted_col[1:])[0]  # (n_boundaries,)
        if len(bi) == 0:
            continue

        n_lefts = bi + 1                        # samples in left child
        n_rights = n - n_lefts                  # samples in right child

        # Enforce min_samples_leaf on both sides.
        valid = (n_lefts >= min_samples_leaf) & (n_rights >= min_samples_leaf)
        if not valid.any():
            continue

        # Left counts at every candidate boundary, read straight out of the
        # cumulative sum; right counts are just "everything else" (total
        # minus left), needing no separate cumulative pass.
        left_counts = cumsum[bi]                    # (n_boundaries, n_classes)
        right_counts = all_counts - left_counts     # (n_boundaries, n_classes)

        # Vectorised impurity for all splits at once.
        left_imp = _batch_imp(left_counts, n_lefts)
        right_imp = _batch_imp(right_counts, n_rights)
        gains = parent_impurity - (
            (n_lefts / n) * left_imp + (n_rights / n) * right_imp
        )

        # Mask out invalid splits.
        # np.where replaces any split that violates min_samples_leaf with
        # -infinity, so it can never win the upcoming argmax.
        gains = np.where(valid, gains, -np.inf)
        best_local = int(np.argmax(gains))

        if gains[best_local] > best_gain:
            best_gain = float(gains[best_local])
            best_feat = feat
            bi_best = bi[best_local]
            # The actual threshold used is the midpoint between the two
            # distinct feature values that straddle this boundary — a
            # sample exactly at the boundary value would need a definite
            # side to fall on, and the midpoint keeps the split symmetric.
            best_thresh = float(
                (sorted_col[bi_best] + sorted_col[bi_best + 1]) / 2.0
            )

    return best_feat, best_thresh, best_gain


# Receives: X_k — only the k candidate feature columns for this node's
#   samples (already selected by the caller, e.g. a random subset for
#   Random Forest), y_node — this node's integer labels, min_samples_leaf,
#   and criterion.
# Calculates: the best split among just these k columns, batching the sort,
#   cumulative-count, and boundary-detection work across all k columns at
#   once (one argsort call, one cumsum call, one boundary-detection call)
#   instead of looping per column for every step.
# Returns: a 5-tuple (best_local_feat, best_threshold, best_gain,
#   parent_impurity, node_majority) — `best_local_feat` is an index into
#   X_k's own columns (0..k-1), not a global feature number, and
#   `parent_impurity`/`node_majority` are returned too since they're
#   already computed here and the caller would otherwise redo that work.
def find_best_split_direct(
    X_k: NDArray,
    y_node: NDArray,
    min_samples_leaf: int,
    criterion: str,
) -> tuple[int | None, float | None, float, float, int]:
    """Find best split on a pre-selected k-feature matrix — used in RF mode.

    information_gain_optimization — caller passes only the k selected columns
    (shape n_node × k), avoiding a copy of all n_features columns.  Returns
    a *local* column index (0..k-1); the caller maps it back to the global
    feature number via ``feature_indices[local_idx]``.

    Uses the same batch strategy as ``find_best_split_presorted``:
    - ONE ``np.argsort(axis=0)`` for all k columns instead of k separate sorts
    - ONE batch cumsum across all k label sequences
    - ONE ``np.where`` call for all boundaries of all features
    - xlogy2 lookup-table (binary path) to replace per-boundary log2 calls

    Parameters
    ----------
    X_k : NDArray of shape (n_node, k)
        Feature values for the k selected features in this node's samples.
    y_node : NDArray of shape (n_node,)
        Integer labels for the current node's samples.
    min_samples_leaf : int
    criterion : str — ``'entropy'`` or ``'gini'``

    Returns
    -------
    best_local_feat : int or None
        Column index within X_k (0..k-1).
    best_threshold : float or None
    best_gain : float
    parent_impurity : float
        Node impurity (already computed here — caller reuses to avoid a
        second entropy/gini call for ``node.impurity``).
    node_majority : int
        Majority-class label (argmax of counts — also already computed here).
    """
    n = len(y_node)
    if n < 2:
        # Nothing to split — return a trivial "no split" result, but still
        # report a majority class (the single sample's own label, or 0 if
        # there isn't even one).
        return None, None, 0.0, 0.0, int(y_node[0]) if n == 1 else 0

    k = X_k.shape[1]
    n_classes = int(y_node.max()) + 1
    all_counts = np.bincount(y_node, minlength=n_classes).astype(np.float64)
    _scalar_imp = _entropy_from_counts if criterion == 'entropy' else _gini_from_counts
    parent_impurity = _scalar_imp(all_counts, n)
    node_majority = int(all_counts.argmax())

    best_feat: int | None = None
    best_thresh: float | None = None
    best_gain: float = 0.0

    # Batch argsort: ONE numpy call for all k columns (vs k separate calls).
    # local_ord[j, i] = local row index of the j-th element when column i is sorted.
    # `axis=0` sorts each of the k columns independently, but all in one
    # call — numpy's argsort supports sorting many columns at once instead
    # of needing a Python loop over columns.
    local_ord: NDArray = np.argsort(X_k, axis=0, kind='quicksort')  # (n_node, k)

    # Build (k, n_node) C-contiguous arrays via direct fancy indexing on local_ord.T.
    # Avoids intermediate (n_node, k) arrays and ascontiguousarray copies.
    # `local_ord.T` transposes to (k, n_node), and `k_range[:, None]` pairs
    # each of the k rows with its own column index, so this fancy-indexing
    # expression gathers, for every feature, its OWN sorted values directly
    # into a (k, n_node) layout — avoiding a separate transpose-and-copy
    # step afterward.
    k_range: NDArray = np.arange(k)   # reused for col_idx and searchsorted below
    X_sg: NDArray = X_k[local_ord.T, k_range[:, None]]  # (k, n_node)
    y_sg: NDArray = y_node[local_ord.T]                  # (k, n_node)

    if n_classes == 2:
        # Binary-class fast path: every count can be tracked as a single
        # "how many positives" number instead of a full class-count vector.
        total_pos = int(all_counts[1])

        # Single batch cumsum for all k label sequences.
        # Running down axis=1 (along each row's own sorted sample order)
        # gives, for every feature and every prefix length, how many
        # positive-class samples are in that prefix.
        cumsums: NDArray = np.cumsum(y_sg, axis=1)  # (k, n_node)

        # Single np.where call for all feature boundaries at once.
        # `X_sg[:, :-1] != X_sg[:, 1:]` compares each row against itself
        # shifted by one column, marking True wherever a feature's sorted
        # value changes from one position to the next — i.e. every valid
        # split boundary, for every feature, all in one comparison.
        diffs: NDArray = X_sg[:, :-1] != X_sg[:, 1:]  # (k, n_node-1)
        feat_arr: NDArray
        bi_arr: NDArray
        # np.where on a 2D array returns two parallel arrays: which row
        # (feature) and which column (boundary position) every True entry
        # came from.
        feat_arr, bi_arr = np.where(diffs)

        if len(bi_arr) > 0:
            n_lefts_all: NDArray = bi_arr + 1
            n_rights_all: NDArray = n - n_lefts_all
            valid_all: NDArray = ((n_lefts_all >= min_samples_leaf)
                                  & (n_rights_all >= min_samples_leaf))
            # For every boundary, look up the already-computed cumulative
            # positive count at that exact (feature, position) pair.
            lp_all: NDArray = cumsums[feat_arr, bi_arr]

            if criterion == 'entropy':
                # xlogy2 table: built once, replaces per-boundary log2 calls.
                # xlogy2[x] = x * log2(x) for every integer x from 0 to n.
                # Because entropy only ever needs this exact "count times
                # log2 of that count" expression, precomputing it once as a
                # lookup table lets every boundary's entropy contribution be
                # read off by simple array indexing instead of calling
                # log2() again and again.
                xlogy2: NDArray = np.empty(n + 1, dtype=np.float64)
                xlogy2[0] = 0.0
                _xs: NDArray = np.arange(1, n + 1, dtype=np.float64)
                xlogy2[1:] = _xs * np.log2(_xs)

                ln_all = n_lefts_all - lp_all
                rp_all = total_pos - lp_all
                rn_all = n_rights_all - rp_all
                # This combination of six table lookups computes the
                # (unnormalised) entropy contribution for every boundary at
                # once — algebraically equivalent to computing each child's
                # binary entropy directly, just rearranged to only need
                # table lookups instead of repeated log2 calls.
                nH_all: NDArray = (
                    xlogy2[n_lefts_all] - xlogy2[lp_all] - xlogy2[ln_all]  # type: ignore[index]
                    + xlogy2[n_rights_all] - xlogy2[rp_all] - xlogy2[rn_all]  # type: ignore[index]
                )
                gains_all: NDArray = np.where(valid_all,
                                              parent_impurity - nH_all / n,
                                              -np.inf)
            else:
                # Gini path: same 2*p*(1-p) simplification used in
                # `_binary_gini_gain`, just computed directly here rather
                # than calling that helper (since the counts are already
                # laid out as flat per-boundary arrays, not the (m, 2)
                # shape that helper expects).
                nl_f: NDArray = n_lefts_all.astype(np.float64)
                nr_f: NDArray = n_rights_all.astype(np.float64)
                lp_f: NDArray = lp_all.astype(np.float64)
                rp_f: NDArray = float(total_pos) - lp_f
                gains_all = np.where(
                    valid_all,
                    parent_impurity - 2.0 * (
                        lp_f * (nl_f - lp_f) / (n * nl_f)
                        + rp_f * (nr_f - rp_f) / (n * nr_f)
                    ),
                    -np.inf,
                )

            # Per-feature argmax: k-iteration loop (not n_node).
            # `feat_arr` is sorted by feature index already (np.where's
            # row-major order guarantees this), so `np.searchsorted` finds,
            # for each feature, where its block of boundaries starts inside
            # the flat `gains_all` array — letting us slice out just that
            # feature's own candidates instead of filtering the whole array
            # k times.
            splits: NDArray = np.searchsorted(feat_arr, k_range)
            total_bi = len(bi_arr)
            for i in range(k):
                seg_start = int(splits[i])
                seg_end = int(splits[i + 1]) if i + 1 < k else total_bi
                if seg_start >= seg_end:
                    continue
                seg = gains_all[seg_start:seg_end]
                local_best = int(seg.argmax())
                gain_val = float(seg[local_best])
                if gain_val > best_gain:
                    best_gain = gain_val
                    best_feat = i       # local column index (0..k-1)
                    pos = seg_start + local_best
                    bi_best = int(bi_arr[pos])
                    best_thresh = float(
                        (X_sg[i, bi_best] + X_sg[i, bi_best + 1]) / 2.0
                    )

    else:
        # Multi-class: per-feature 2-D cumsum (k-iteration loop).
        # With more than 2 classes, the compact "positive count" trick
        # above no longer applies, so this path falls back to a small loop
        # over the k features (still far fewer than n_node samples) using
        # the general multi-class impurity helpers.
        _batch_imp = _batch_entropy if criterion == 'entropy' else _batch_gini
        diffs_mc: NDArray = X_sg[:, :-1] != X_sg[:, 1:]

        for i in range(k):
            if not diffs_mc[i].any():
                # This feature's values never change in sorted order for
                # this node — every sample has the same value, so there's
                # no split to consider.
                continue
            bi = np.where(diffs_mc[i])[0]
            n_lefts = bi + 1
            n_rights = n - n_lefts
            valid = (n_lefts >= min_samples_leaf) & (n_rights >= min_samples_leaf)
            if not valid.any():
                continue

            # One-hot encode this feature's own sorted labels, then
            # cumulative-sum down the sample axis to get left-child class
            # counts at every position — same idea as `find_best_split`,
            # just done once per feature here instead of once for the whole
            # node.
            y_oh: NDArray = (y_sg[i][:, None] == np.arange(n_classes)).astype(np.float64)
            cumsum = np.cumsum(y_oh, axis=0)
            left_counts = cumsum[bi]
            right_counts = all_counts - left_counts
            left_imp = _batch_imp(left_counts, n_lefts.astype(np.float64))
            right_imp = _batch_imp(right_counts, n_rights.astype(np.float64))
            gains = parent_impurity - (n_lefts / n * left_imp + n_rights / n * right_imp)
            gains = np.where(valid, gains, -np.inf)
            local_best = int(gains.argmax())
            if gains[local_best] > best_gain:
                best_gain = float(gains[local_best])
                best_feat = i           # local column index (0..k-1)
                bi_mc = int(bi[local_best])
                best_thresh = float(
                    (X_sg[i, bi_mc] + X_sg[i, bi_mc + 1]) / 2.0
                )

    return best_feat, best_thresh, best_gain, parent_impurity, node_majority


# Receives: X — the FULL training matrix (never subsetted to just this
#   node), y — the full label array, sort_indices — a precomputed table
#   where sort_indices[f] already lists this node's own sample indices in
#   feature-f-sorted order, feature_indices — which columns to consider,
#   min_samples_leaf, and criterion.
# Calculates: the best split among the given feature_indices, reusing the
#   already-sorted order from `sort_indices` (no re-sorting needed) and
#   batching the gather/cumsum/boundary-detection work across every
#   requested feature at once.
# Returns: a 3-tuple (best_feature, best_threshold, best_gain) — here
#   `best_feature` is a GLOBAL column index into X (unlike
#   `find_best_split_direct`'s local index), since this function receives
#   the full X and a list of candidate columns rather than a pre-sliced
#   sub-matrix.
def find_best_split_presorted(
    X: NDArray,
    y: NDArray,
    sort_indices: NDArray,
    feature_indices: NDArray,
    min_samples_leaf: int,
    criterion: str,
) -> tuple[int | None, float | None, float]:
    """Find the best binary split using pre-sorted per-feature indices.

    information_gain_optimization — three-level speedup over re-sorting:

    1. **No argsort**: ``sort_indices[f]`` already gives this node's samples
       sorted by feature f, built once in ``fit()`` and partitioned through
       the recursion.  Per-feature work drops from O(n log n) to O(n).

    2. **Batch gather + batch cumsum**: both the feature-value gather and the
       label cumsum are issued as two NumPy calls covering *all* selected
       features at once, halving Python/NumPy call overhead relative to a
       per-feature loop.

    3. **xlogy2 lookup table** (binary entropy path): replaces per-boundary
       ``log2`` calls with O(1) integer table lookups; the table ``xlogy2[x]
       = x * log2(x)`` is built once per node call (O(n) integers) and
       reused for all features, collapsing 20+ ``log2`` calls to 6 index
       reads per boundary set.

    Parameters
    ----------
    X : NDArray of shape (n_total, n_features)
        Full training matrix (never subsetted).
    y : NDArray of shape (n_total,)
        Full integer label array (never subsetted).
    sort_indices : NDArray of shape (n_features, n_node)
        For each feature ``f``, ``sort_indices[f]`` contains this node's
        original sample indices sorted by ``X[:, f]``.
    feature_indices : NDArray of int
        Feature columns to consider for splitting.
    min_samples_leaf : int
    criterion : str — ``'entropy'`` or ``'gini'``

    Returns
    -------
    best_feature : int or None
    best_threshold : float or None
    best_gain : float
    """
    n = sort_indices.shape[1]
    if n < 2:
        return None, None, 0.0

    # Batch-gather: 2 NumPy calls cover ALL selected features at once.
    # si[i] = sort_indices[feature_indices[i]] — node rows sorted by feature i.
    # Selecting just the requested rows of `sort_indices` (one row per
    # candidate feature) in a single fancy-index call avoids looping over
    # `feature_indices` in Python just to gather each one individually.
    si: NDArray = sort_indices[feature_indices]             # (k, n_node)
    # `feature_indices[:, None]` pairs each selected feature's row in `si`
    # with its OWN column in X, so this expression gathers every feature's
    # own already-sorted feature values directly into a (k, n_node) table.
    X_sg: NDArray = X[si, feature_indices[:, None]]        # (k, n_node) feature values
    y_sg: NDArray = y[si]                                   # (k, n_node) labels

    # Any one feature's sorted label order describes the same node, so its
    # first row's labels stand in for "this node's labels" when counting
    # classes overall.
    y_node = y_sg[0]
    n_classes = int(y_node.max()) + 1
    all_counts = np.bincount(y_node, minlength=n_classes).astype(np.float64)

    _scalar_imp = _entropy_from_counts if criterion == 'entropy' else _gini_from_counts
    parent_impurity = _scalar_imp(all_counts, n)

    best_feat: int | None = None
    best_thresh: float | None = None
    best_gain: float = 0.0

    if n_classes == 2:
        total_pos = int(all_counts[1])

        # Batch cumsum: one call for all k features (replaces k individual cumsums).
        cumsums: NDArray = np.cumsum(y_sg, axis=1)  # (k, n_node)

        # Batch boundary detection: one np.where call for all k features at once.
        # diffs[i, j] = True where feature i has a value change between positions j and j+1.
        diffs: NDArray = X_sg[:, :-1] != X_sg[:, 1:]  # (k, n_node-1) bool

        # Collect ALL boundary positions for ALL features in one np.where call.
        # feat_arr[b]: which feature (row in si); bi_arr[b]: position in sorted order.
        # np.where on a 2-D array returns in row-major order, so feat_arr is non-decreasing.
        feat_arr: NDArray
        bi_arr: NDArray
        feat_arr, bi_arr = np.where(diffs)

        if len(bi_arr) > 0:
            # Vectorised gain computation for EVERY boundary of EVERY feature at once —
            # no Python loop over boundaries.  The per-feature loop below runs only k
            # times (k = |feature_indices|, usually ≤ 20 or sqrt(20) for RF).
            n_lefts_all: NDArray = bi_arr + 1
            n_rights_all: NDArray = n - n_lefts_all
            valid_all: NDArray = ((n_lefts_all >= min_samples_leaf)
                                  & (n_rights_all >= min_samples_leaf))

            lp_all: NDArray = cumsums[feat_arr, bi_arr]   # left class-1 count

            if criterion == 'entropy':
                # xlogy2 lookup table: xlogy2[x] = x * log2(x), xlogy2[0] = 0.
                # Built once per node, shared across ALL features.
                # Replaces per-boundary log2() calls with integer table lookups.
                xlogy2: NDArray = np.empty(n + 1, dtype=np.float64)
                xlogy2[0] = 0.0
                _xs: NDArray = np.arange(1, n + 1, dtype=np.float64)
                xlogy2[1:] = _xs * np.log2(_xs)

                ln_all = n_lefts_all - lp_all
                rp_all = total_pos - lp_all
                rn_all = n_rights_all - rp_all
                nH_all: NDArray = (
                    xlogy2[n_lefts_all] - xlogy2[lp_all] - xlogy2[ln_all]  # type: ignore[index]
                    + xlogy2[n_rights_all] - xlogy2[rp_all] - xlogy2[rn_all]  # type: ignore[index]
                )
                gains_all: NDArray = np.where(valid_all,
                                              parent_impurity - nH_all / n,
                                              -np.inf)
            else:
                # Gini: 2*lp*(n_left - lp) / (n * n_left) + 2*rp*(n_right - rp) / (n * n_right)
                nl_f: NDArray = n_lefts_all.astype(np.float64)
                nr_f: NDArray = n_rights_all.astype(np.float64)
                lp_f: NDArray = lp_all.astype(np.float64)
                rp_f: NDArray = (float(total_pos) - lp_f)
                gains_all = np.where(
                    valid_all,
                    parent_impurity - 2.0 * (
                        lp_f * (nl_f - lp_f) / (n * nl_f)
                        + rp_f * (nr_f - rp_f) / (n * nr_f)
                    ),
                    -np.inf,
                )

            # Per-feature best: loop of k iterations (not n_node boundaries).
            # splits[i] = first index in feat_arr/bi_arr for feature i.
            k = len(feature_indices)
            splits: NDArray = np.searchsorted(feat_arr, np.arange(k))
            total_bi = len(bi_arr)

            for i in range(k):
                seg_start = int(splits[i])
                seg_end = int(splits[i + 1]) if i + 1 < k else total_bi
                if seg_start >= seg_end:
                    continue
                seg = gains_all[seg_start:seg_end]
                local_best = int(seg.argmax())
                gain_val = float(seg[local_best])
                if gain_val > best_gain:
                    best_gain = gain_val
                    # Unlike find_best_split_direct, `i` here maps straight
                    # back to a real column of X via feature_indices, since
                    # this function was handed the full feature list, not a
                    # pre-sliced sub-matrix.
                    best_feat = int(feature_indices[i])
                    pos = seg_start + local_best
                    bi_best = int(bi_arr[pos])
                    best_thresh = float(
                        (X_sg[i, bi_best] + X_sg[i, bi_best + 1]) / 2.0
                    )

    else:
        # Multi-class path — per-feature 2-D cumsum (correctness over speed for n_classes > 2)
        _batch_imp = _batch_entropy if criterion == 'entropy' else _batch_gini
        diffs_mc: NDArray = X_sg[:, :-1] != X_sg[:, 1:]  # (k, n_node-1)

        for i, feat in enumerate(feature_indices):
            if not diffs_mc[i].any():
                continue
            bi = np.where(diffs_mc[i])[0]
            n_lefts = bi + 1
            n_rights = n - n_lefts
            valid = (n_lefts >= min_samples_leaf) & (n_rights >= min_samples_leaf)
            if not valid.any():
                continue

            y_oh: NDArray = (y_sg[i][:, None] == np.arange(n_classes)).astype(np.float64)
            cumsum = np.cumsum(y_oh, axis=0)
            left_counts = cumsum[bi]
            right_counts = all_counts - left_counts
            left_imp = _batch_imp(left_counts, n_lefts.astype(np.float64))
            right_imp = _batch_imp(right_counts, n_rights.astype(np.float64))
            gains = parent_impurity - (n_lefts / n * left_imp + n_rights / n * right_imp)
            gains = np.where(valid, gains, -np.inf)
            best_local = int(gains.argmax())
            if gains[best_local] > best_gain:
                best_gain = float(gains[best_local])
                best_feat = int(feat)
                bi_best = int(bi[best_local])
                sorted_col_mc: NDArray = X_sg[i]
                best_thresh = float((sorted_col_mc[bi_best] + sorted_col_mc[bi_best + 1]) / 2.0)

    return best_feat, best_thresh, best_gain
