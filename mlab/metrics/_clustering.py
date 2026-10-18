"""Clustering quality metrics for KMeans / KMeansPlusPlus.

`silhouette_score` and `davies_bouldin_score` don't need ground-truth
labels — they judge a clustering purely from the points and the cluster
assignments it produced, which is what you're stuck with in real
unsupervised use. `KMeans`/`KMeansPlusPlus` already track `inertia_`
(within-cluster sum of squares) themselves, so it isn't repeated here.

`adjusted_rand_score` is different: it needs a *known* ground-truth
labelling to compare against, which only exists for synthetic/benchmark
data (e.g. `sklearn.datasets.make_blobs` hands back the true cluster each
point was generated from). It's still a clustering metric, not a
classification one, because it scores agreement between two groupings up
to a relabelling — cluster "0" in one run and cluster "2" in another still
count as a perfect match if they contain the same points.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray


# Receives: `X` — array-like feature data (should be a table of samples by
#   features).
# Calculates: converts `X` to a numpy array of 64-bit floats and checks that
#   it really is 2-dimensional (rows = samples, columns = features).
# Returns: the validated 2-D float64 array; raises ValueError if `X` isn't
#   2-dimensional.
def _as_feature_matrix(X: Union[Sequence, NDArray]) -> NDArray:
    # Force whatever was passed in (a list of lists, a pandas DataFrame's
    # underlying array, etc.) into a plain numpy array of floats.
    arr = np.asarray(X, dtype=np.float64)
    # Clustering data must be a 2D table: one row per sample, one column per
    # feature. `arr.ndim` is how many dimensions/axes the array has (1 for a
    # flat list of numbers, 2 for a table, 3 for a stack of tables, etc.).
    # If someone passes a flat 1D array by mistake, fail loudly here instead
    # of producing confusing errors deep inside the distance math below.
    if arr.ndim != 2:
        raise ValueError(f"X must be a 2D (n_samples, n_features) array, got shape {arr.shape}")
    return arr


# Receives: `a` and `b` — two 2-D arrays of points (each row is one point).
# Calculates: the straight-line (Euclidean) distance between every point in
#   `a` and every point in `b`, all at once via broadcasting.
# Returns: a 2-D array of shape (len(a), len(b)) where entry [i, j] is the
#   distance between point i of `a` and point j of `b`.
def _pairwise_euclidean(a: NDArray, b: NDArray) -> NDArray:
    """Euclidean distance between every row of `a` and every row of `b`,
    via the expanded-form broadcast rather than a python double loop."""
    # `a` has shape (n_a, n_features) and `b` has shape (n_b, n_features).
    # `a[:, None, :]` inserts a new axis in the middle, reshaping `a` to
    # (n_a, 1, n_features); `b[None, :, :]` reshapes `b` to (1, n_b,
    # n_features). When numpy subtracts arrays of shapes (n_a, 1, F) and
    # (1, n_b, F), "broadcasting" stretches each size-1 axis to match the
    # other array, producing every possible pairing: the result has shape
    # (n_a, n_b, F), where diff[i, j] is the feature-by-feature difference
    # between row i of `a` and row j of `b`. This computes ALL pairwise
    # differences in one vectorised step, instead of looping over i and j.
    diff = a[:, None, :] - b[None, :, :]
    # Squaring each difference and summing over the last axis (the features)
    # collapses (n_a, n_b, F) down to (n_a, n_b): one squared-distance number
    # per (row of a, row of b) pair.
    sq_dist = np.sum(diff ** 2, axis=-1)
    # In exact arithmetic sq_dist can never be negative, but floating-point
    # rounding can occasionally push a "should be zero" distance very
    # slightly below zero. np.maximum(..., 0.0) clamps that away before the
    # square root, so we never take sqrt of a negative number.
    return np.sqrt(np.maximum(sq_dist, 0.0))


# Receives: `X` — the feature data that was clustered, and `labels` — which
#   cluster each row of `X` was assigned to.
# Calculates: for every point, how much closer it is (on average) to the
#   other points in its own cluster than to the points in the nearest other
#   cluster, then averages that value across every point.
# Returns: a single float between -1 and 1 (higher means better-separated,
#   more internally-consistent clusters).
def silhouette_score(X: Union[Sequence, NDArray], labels: Union[Sequence, NDArray]) -> float:
    """Mean Silhouette Coefficient over every sample: for each point, how
    much closer it is (on average) to its own cluster than to the nearest
    other cluster. Ranges from -1 (probably in the wrong cluster) to +1
    (well matched to its own cluster and far from the others); values near
    0 mean overlapping clusters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    labels : array-like of shape (n_samples,)
        Cluster assignment for each sample (e.g. `KMeans.labels_`).

    Returns
    -------
    float in [-1, 1]
    """
    X = _as_feature_matrix(X)
    labels = np.asarray(labels)
    n_samples = X.shape[0]
    if labels.shape[0] != n_samples:
        raise ValueError(f"X has {n_samples} samples but labels has {labels.shape[0]}")

    # np.unique returns every distinct cluster id that appears in `labels`,
    # sorted, with duplicates removed — e.g. labels=[0,0,1,2,1] gives [0,1,2].
    unique_labels = np.unique(labels)
    n_labels = unique_labels.shape[0]
    # The silhouette score is only mathematically defined when there are at
    # least 2 clusters (otherwise "distance to the nearest OTHER cluster"
    # doesn't exist) and at most n_samples - 1 clusters (otherwise every
    # point would be alone in its own cluster).
    if not 2 <= n_labels <= n_samples - 1:
        raise ValueError(
            "silhouette_score needs between 2 and n_samples - 1 clusters, "
            f"got {n_labels} clusters for {n_samples} samples"
        )

    # One-time upfront cost: compute the distance between every pair of
    # points in the dataset. `dist[i, j]` is now the distance from point i
    # to point j, for every i and j.
    dist = _pairwise_euclidean(X, X)
    # We'll fill in one silhouette value per sample as we go, cluster by
    # cluster.
    scores = np.zeros(n_samples)

    for label in unique_labels:
        # `labels == label` compares every entry of `labels` against this
        # cluster id and produces a boolean array, e.g. [True, False, True,
        # ...]. Using that boolean array to index into another array (a
        # "boolean mask") selects only the entries where it's True — this
        # is how we grab "just the points belonging to this cluster".
        in_cluster = labels == label
        cluster_size = int(in_cluster.sum())  # True counts as 1, False as 0
        if cluster_size == 1:
            # a lone point has no "intra-cluster distance" to compare against,
            # so scikit-learn defines its silhouette as exactly 0
            scores[in_cluster] = 0.0
            continue

        # a: mean distance to every other point in the same cluster.
        # np.ix_(in_cluster, in_cluster) picks out the sub-block of the full
        # distance matrix containing only rows AND columns for this
        # cluster's points — i.e. every pairwise distance *within* the
        # cluster. Summing each row and dividing by (cluster_size - 1)
        # gives, per point, its average distance to every other member of
        # its own cluster (we subtract 1 because a point's distance to
        # itself is always 0 and shouldn't count).
        intra = dist[np.ix_(in_cluster, in_cluster)].sum(axis=1) / (cluster_size - 1)

        # b: mean distance to each other cluster, keeping only the closest one.
        # Start by assuming every point is infinitely far from "the nearest
        # other cluster", then update that as we check each candidate cluster.
        nearest_other = np.full(cluster_size, np.inf)
        for other_label in unique_labels:
            if other_label == label:
                continue  # skip comparing the cluster to itself
            other_mask = labels == other_label
            # This time we want a rectangular block: rows = this cluster's
            # points, columns = the OTHER cluster's points. Averaging along
            # axis=1 (across columns) gives, per point in our cluster, its
            # mean distance to every point in that other cluster.
            mean_dist_to_other = dist[np.ix_(in_cluster, other_mask)].mean(axis=1)
            # np.minimum keeps whichever is smaller, element by element. After
            # looping over every other cluster, `nearest_other` holds each
            # point's distance to the single closest *other* cluster.
            nearest_other = np.minimum(nearest_other, mean_dist_to_other)

        # The silhouette formula is (b - a) / max(a, b) per point.
        denom = np.maximum(intra, nearest_other)
        # np.where(condition, value_if_true, value_if_false) picks between
        # two arrays element by element based on a boolean condition — here,
        # avoid a 0/0 division for the (extremely rare) case where both `a`
        # and `b` are exactly 0 for some point.
        cluster_scores = np.where(denom > 0, (nearest_other - intra) / denom, 0.0)
        scores[in_cluster] = cluster_scores

    # The overall silhouette score reported for the whole clustering is just
    # the plain average of every individual point's score.
    return float(scores.mean())


# Receives: `X` — the feature data that was clustered, and `labels` — which
#   cluster each row of `X` was assigned to.
# Calculates: for every cluster, its "similarity" to the single other
#   cluster it most resembles (large when both clusters are spread-out and
#   close together, small when compact and far apart), then averages that
#   worst-case similarity across every cluster.
# Returns: a single float >= 0 (lower means better-separated clusters; 0 is
#   the best possible score).
def davies_bouldin_score(X: Union[Sequence, NDArray], labels: Union[Sequence, NDArray]) -> float:
    """Average, over every cluster, of its similarity to the single other
    cluster it resembles most — where "similarity" is how large the two
    clusters' own spread is relative to how far apart their centroids are.
    Lower is better (0 is the best possible score); unlike the silhouette
    score there's no upper bound.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    labels : array-like of shape (n_samples,)
        Cluster assignment for each sample (e.g. `KMeans.labels_`).

    Returns
    -------
    float >= 0
    """
    X = _as_feature_matrix(X)
    labels = np.asarray(labels)
    if labels.shape[0] != X.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples but labels has {labels.shape[0]}")

    unique_labels = np.unique(labels)
    n_clusters = unique_labels.shape[0]
    if n_clusters < 2:
        raise ValueError(f"davies_bouldin_score needs at least 2 clusters, got {n_clusters}")

    # `centroids` will hold one row per cluster: the "center of mass" of
    # that cluster's points, computed as the plain average of their
    # coordinates.
    centroids = np.zeros((n_clusters, X.shape[1]))
    # intra_dist[k]: how spread out cluster k is, as the mean distance of its
    # own points to its own centroid (not an RMS/std — a plain average, to
    # match the metric's original definition)
    intra_dist = np.zeros(n_clusters)
    # enumerate(unique_labels) walks through the cluster ids while also
    # giving us a plain counting index `i` (0, 1, 2, ...) to store results at.
    for i, label in enumerate(unique_labels):
        # boolean-mask indexing again: grab only the rows of X belonging to
        # this cluster.
        members = X[labels == label]
        # `.mean(axis=0)` averages down each column (each feature)
        # separately, giving one number per feature — i.e. the centroid.
        centroids[i] = members.mean(axis=0)
        # Distance from every member point to this one centroid, then
        # averaged into a single "how spread out is this cluster?" number.
        intra_dist[i] = _pairwise_euclidean(members, centroids[i:i + 1]).mean()

    # All pairwise distances between cluster centroids at once (same
    # broadcasting trick as `_pairwise_euclidean` used above, just applied
    # to the small (n_clusters, n_features) centroid array instead of the
    # full dataset).
    centroid_dist = _pairwise_euclidean(centroids, centroids)

    # degenerate clustering: every cluster is a single point (no spread), or
    # every centroid landed on top of every other one — neither case has a
    # meaningful separation to score, so call it a perfect 0 rather than
    # dividing by zero
    if np.allclose(intra_dist, 0.0) or np.allclose(centroid_dist, 0.0):
        return 0.0

    # the diagonal is a cluster measured against itself, which is always
    # zero — sending those entries to +inf makes them vanish from the max()
    # below instead of needing a separate diagonal mask
    centroid_dist = np.where(centroid_dist == 0.0, np.inf, centroid_dist)
    # `intra_dist[:, None] + intra_dist[None, :]` uses the same broadcasting
    # idea as before: it adds every pair of clusters' spreads together,
    # producing an (n_clusters, n_clusters) matrix where entry [i, j] is
    # "how spread out is cluster i, plus how spread out is cluster j".
    combined_spread = intra_dist[:, None] + intra_dist[None, :]
    # Dividing that by how far apart the two centroids are gives a
    # "similarity" score for every pair of clusters: big when the clusters
    # are both spread-out AND close together (bad — they're hard to tell
    # apart), small when they're compact and far apart (good). For each
    # cluster (each row), we keep only its worst (largest) similarity to
    # any other cluster — its "nearest rival".
    worst_match = np.max(combined_spread / centroid_dist, axis=1)
    # The reported score is the average of every cluster's worst-case
    # similarity to its most-alike neighbour.
    return float(worst_match.mean())


# Receives: `labels_true` and `labels_pred` — two label arrays for the same
#   samples (e.g. the true generating cluster, and the predicted cluster).
# Calculates: how many samples fall into each possible (true label,
#   predicted label) combination.
# Returns: a 2-D array of counts, shape (n_true_labels, n_pred_labels) —
#   entry [i, j] is how many samples had true label i AND predicted label j.
def _contingency_matrix(labels_true: NDArray, labels_pred: NDArray) -> NDArray:
    """How many samples fall into each (true cluster, predicted cluster)
    pair — the two-grouping analogue of `_classification.confusion_matrix`,
    except the row and column label sets don't need to match (a predicted
    cluster is just an arbitrary group id, not "the same" class as any
    particular true label)."""
    # return_inverse=True gives us, alongside the sorted list of distinct
    # labels, an array the same length as the input where every original
    # label has been replaced by "which position in the distinct list is
    # this?" — e.g. labels_true=['a','b','a'] with distinct=['a','b'] gives
    # true_idx=[0,1,0]. This turns arbitrary label values (strings, mixed
    # types, whatever) into plain 0-based integer indices we can do
    # arithmetic with.
    true_ids, true_idx = np.unique(labels_true, return_inverse=True)
    pred_ids, pred_idx = np.unique(labels_pred, return_inverse=True)
    n_true, n_pred = true_ids.shape[0], pred_ids.shape[0]
    # Combine each sample's (true group index, predicted group index) pair
    # into one single flat index, using the standard "row * n_columns +
    # column" trick for addressing a 2D grid with a 1D array. This lets us
    # count how many samples land in each (true, predicted) combination
    # using a single fast np.bincount call instead of a nested loop.
    flat_idx = true_idx * n_pred + pred_idx
    # np.bincount counts how many times each integer 0, 1, 2, ... appears in
    # flat_idx. `minlength` makes sure the output has exactly n_true * n_pred
    # slots even if some (true, predicted) combination never occurs.
    # `.reshape(n_true, n_pred)` folds that flat list of counts back into a
    # proper 2D table: row = true group, column = predicted group.
    return np.bincount(flat_idx, minlength=n_true * n_pred).reshape(n_true, n_pred)


# Receives: `counts` — an array where each entry is "how many samples belong
#   to this one group".
# Calculates: for each entry, "n choose 2" — the number of distinct pairs
#   you can form by picking 2 samples out of that group.
# Returns: an array the same shape as `counts`, holding the pair count for
#   each group.
def _n_choose_2(counts: NDArray) -> NDArray:
    """sum of C(n, 2) for each count n — "how many same-group pairs can be
    formed from this many points", the building block the Rand index is
    defined in terms of."""
    # "n choose 2" is the number of distinct ways to pick 2 items out of a
    # group of n, with order not mattering — e.g. choosing 2 friends out of
    # a group of 4 to invite. Its formula is n * (n - 1) / 2.
    counts = counts.astype(np.float64)
    return counts * (counts - 1.0) / 2.0


# Receives: `labels_true` — the known, correct cluster/class for each
#   sample, and `labels_pred` — the cluster assignment a model actually
#   produced for those same samples.
# Calculates: how many pairs of samples the two groupings agree on (both
#   put them together, or both keep them apart), corrected for how much
#   agreement would happen purely by chance given the group sizes involved.
# Returns: a single float, <= 1.0 — 1.0 means the two groupings match
#   perfectly (up to renaming the clusters), 0.0 is what pure chance would
#   score on average, and it can go negative for a worse-than-chance match.
def adjusted_rand_score(
    labels_true: Union[Sequence, NDArray], labels_pred: Union[Sequence, NDArray]
) -> float:
    """How well a clustering matches a known ground-truth grouping, corrected
    for the agreement you'd expect from pure chance (an unadjusted match
    rate keeps climbing just from having more clusters, even when the
    clustering is meaningless — this doesn't). 1.0 means identical
    groupings up to relabelling, 0.0 is what random guessing scores on
    average, and it can go negative for a worse-than-random split.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        The ground-truth cluster/class each sample actually belongs to.
    labels_pred : array-like of shape (n_samples,)
        Cluster assignment produced by the model (e.g. `KMeans.labels_`).

    Returns
    -------
    float, <= 1.0
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            f"labels_true and labels_pred must have the same shape, "
            f"got {labels_true.shape} and {labels_pred.shape}"
        )
    n_samples = labels_true.shape[0]
    if n_samples == 0:
        raise ValueError("cannot score adjusted_rand_score on empty input")

    # Build the (true cluster) x (predicted cluster) count table described
    # above, then work out four quantities the Adjusted Rand Index formula
    # is built from. Every one of these is "how many PAIRS of samples end up
    # grouped together", counted a different way:
    contingency = _contingency_matrix(labels_true, labels_pred)
    # pairs of samples that share BOTH their true group and predicted group
    # (i.e. genuinely agree) — sum _n_choose_2 over every cell of the table.
    pair_agreements = _n_choose_2(contingency).sum()
    # pairs of samples that share their true group, ignoring what they were
    # predicted as — sum _n_choose_2 over each row's total (contingency
    # summed across columns, axis=1).
    true_group_pairs = _n_choose_2(contingency.sum(axis=1)).sum()
    # pairs of samples that share their predicted group, ignoring the truth
    # — same idea, but summed down each column (axis=0) instead.
    pred_group_pairs = _n_choose_2(contingency.sum(axis=0)).sum()
    # total number of possible sample pairs in the whole dataset.
    total_pairs = _n_choose_2(np.array([n_samples]))[0]

    # "Expected" here means: how many pair-agreements would we see on
    # average from two *random*, unrelated groupings with these same group
    # sizes? This is the baseline the score is adjusted against.
    expected_agreements = (
        true_group_pairs * pred_group_pairs / total_pairs if total_pairs else 0.0
    )
    # The best possible (maximum) number of agreeing pairs achievable given
    # these group sizes, averaged between the two groupings' own totals.
    max_agreements = 0.5 * (true_group_pairs + pred_group_pairs)
    denominator = max_agreements - expected_agreements

    if denominator == 0.0:
        # both groupings are a single cluster (or otherwise can't disagree at
        # all given their sizes) — perfect agreement by definition
        return 1.0
    # The Adjusted Rand Index formula: (actual agreement - expected chance
    # agreement) / (best possible agreement - expected chance agreement).
    # This is exactly why a "1.0" means perfect (actual = best possible) and
    # "0.0" means no better than chance (actual = expected).
    return float((pair_agreements - expected_agreements) / denominator)
