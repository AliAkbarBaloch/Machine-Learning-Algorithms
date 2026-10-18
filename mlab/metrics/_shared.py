"""Two small helpers shared across the metrics modules.

`check_same_length` is used by both `_classification.py` and
`_regression.py`, which both need "same shape, not empty" before touching
y_true/y_pred, just with a different noun in the error message.

`rank` is used by `_regression.py` (Spearman correlation, which is just
Pearson correlation computed on ranks) and `_classification.py` (ROC-AUC,
which reduces to a rank-sum statistic). Neither module owns the concept, so
it lives here instead of being copy-pasted into whichever file happened to
need it first.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Receives: two arrays, y_true and y_pred, plus a plain string metric_kind
#   ("classification" or "regression") used only to word the error message.
# Calculates: whether y_true and y_pred have the same shape, and whether
#   they contain at least one element.
# Returns: nothing on success (None) — raises ValueError immediately if
#   either check fails, so the caller never has to check a return value.
def check_same_length(y_true: NDArray, y_pred: NDArray, metric_kind: str) -> None:
    """Raise a clear ValueError if `y_true`/`y_pred` can't be compared.

    Parameters
    ----------
    y_true : NDArray
    y_pred : NDArray
    metric_kind : str
        Used only to word the empty-input error, e.g. "classification" or
        "regression".
    """
    # `.shape` is a tuple like (10,) telling you how many numbers are in the
    # array along each dimension. If y_true has 10 values and y_pred has 8,
    # there's no sensible way to compare them one-to-one, so stop here.
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}"
        )
    # `.size` is the total count of numbers in the array. If it's 0, the
    # caller passed in empty arrays — every metric below would either divide
    # by zero or silently return a meaningless number, so we refuse instead.
    if y_true.size == 0:
        raise ValueError(f"cannot score {metric_kind} metrics on empty input")


# Receives: a 1-D array of numeric values (`values`).
# Calculates: the position each value would occupy if the array were sorted
#   from lowest to highest — the smallest value gets rank 1, the next
#   smallest gets rank 2, and so on — with tied values sharing the average
#   of the ranks they'd otherwise span (e.g. two values tied for 2nd/3rd
#   place both get rank 2.5).
# Returns: a new float64 array the same length and in the same order as
#   `values`, holding each element's rank.
def rank(values: NDArray) -> NDArray:
    """Rank values from lowest (rank 1) to highest, giving tied values the
    average of the ranks they'd otherwise span — e.g. [10, 20, 20, 30] ->
    [1, 2.5, 2.5, 4]. This is the standard "fractional ranking" scheme
    Spearman correlation and rank-sum AUC are both defined in terms of.

    Parameters
    ----------
    values : NDArray of shape (n,)

    Returns
    -------
    NDArray of shape (n,), dtype float64
    """
    # np.argsort doesn't sort the array itself — it returns the *indices*
    # that would put the array in sorted order. Example: values = [30, 10, 20]
    # gives sorter = [1, 2, 0], because values[1]=10 is smallest, then
    # values[2]=20, then values[0]=30.
    # kind="mergesort" is used instead of the default "quicksort" because
    # mergesort is *stable*: equal values keep their original relative order,
    # which makes the tie-handling below predictable and testable.
    sorter = np.argsort(values, kind="mergesort")

    # Actually reorder the values into ascending order using those indices.
    # Continuing the example above: sorted_values = [10, 20, 30].
    sorted_values = values[sorter]

    # group identical values together and give every member of a group the
    # same (averaged) rank, rather than an arbitrary tie-break order.
    #
    # np.unique with return_index/return_counts gives us three parallel arrays:
    #   unique_values -> every distinct value, in ascending order
    #   first_pos     -> for each unique value, the index of its FIRST
    #                     occurrence inside sorted_values
    #   counts        -> for each unique value, how many times it repeats
    #
    # Example: sorted_values = [10, 20, 20, 30]
    #   unique_values = [10, 20, 30]
    #   first_pos     = [0,  1,  3]   (20 first shows up at position 1)
    #   counts        = [1,  2,  1]   (20 appears twice)
    unique_values, first_pos, counts = np.unique(
        sorted_values, return_index=True, return_counts=True
    )

    # Ranks in this scheme are 1-based (the smallest value gets rank 1, not
    # rank 0). A group that starts at position `first_pos` and spans `counts`
    # positions occupies 1-based ranks first_pos+1 .. first_pos+counts.
    # The *average* of that consecutive run of integers is:
    #   first_pos + (counts - 1) / 2 + 1
    # Continuing the example: the "20" group spans ranks 2 and 3, so its
    # average rank is (2 + 3) / 2 = 2.5, which is exactly what this formula
    # produces: 1 + (2 - 1) / 2 + 1 = 2.5.
    group_avg_rank = first_pos + (counts - 1) / 2.0 + 1.0

    # We now know the average rank *per distinct value*, but we need it
    # *per original sample*. np.searchsorted looks up, for every entry in
    # sorted_values, which slot in unique_values it matches — i.e. which
    # group (0th, 1st, 2nd, ...) each sample belongs to.
    group_of_each_sample = np.searchsorted(unique_values, sorted_values)

    # group_avg_rank[group_of_each_sample] expands the per-group ranks back
    # out to one rank per sample, still in *sorted* order. The last step is
    # to undo the earlier sort: `ranks[sorter] = ...` scatters each computed
    # rank back to the original (pre-sort) position it came from, so the
    # output lines up with the input `values` array exactly.
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[sorter] = group_avg_rank[group_of_each_sample]
    return ranks
