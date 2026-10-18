"""Classification scoring metrics shared across every classifier in this
library (LogisticRegression, SGDClassifier, GaussianNB, MultinomialNB,
DecisionTreeClassifier, RandomForestClassifier, SVM, SVC, CNNClassifier).

Everything here is built on top of one confusion matrix, since precision,
recall and F1 are all just different ratios of the same true/false
positive and negative counts. There was no reason to scan the predictions
three separate times when one pass gives you everything.

`roc_auc_score` and `average_precision_score` are the odd ones out — they
score predicted *probabilities* (`predict_proba`), not the final labels, so
they don't fit the confusion-matrix machinery above. They apply to every
classifier here that exposes `predict_proba` (LogisticRegression,
SGDClassifier, GaussianNB, MultinomialNB, DecisionTreeClassifier,
RandomForestClassifier, CNNClassifier) — useful whenever the decision
threshold itself is still up for debate, e.g. under class imbalance where a
single accuracy/F1 number at the default 0.5 threshold hides how the model
behaves at every other threshold.

Both accept either a binary `y_score` (1-D, one probability per sample, for
the positive class) or a multiclass `y_score` (2-D, one probability column
per class, e.g. the full output of `predict_proba`). The multiclass case is
scored One-vs-Rest: each class takes a turn as "positive" against every
other class as "negative", using that class's own probability column, and
the resulting per-class scores are combined with `average='macro'`
(unweighted mean) or `average='weighted'` (weighted by class support).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from mlab.metrics._shared import check_same_length, rank


# Receives: `y` — a list or array of class labels, possibly shaped as a
#   column vector instead of a flat list.
# Calculates: converts `y` into a plain numpy array, flattening it to 1-D if
#   it came in as a 2-D column vector.
# Returns: a flat (n,) numpy array of labels (dtype unchanged — labels can
#   be ints, strings, etc.).
def _as_label_array(y: Union[Sequence, NDArray]) -> NDArray:
    """Coerce a list/array of labels into a flat 1-D numpy array."""
    # Accept plain Python lists as well as numpy arrays by normalising
    # everything to an array first.
    arr = np.asarray(y)
    # If someone accidentally passes a column vector of shape (n, 1),
    # flatten it down to (n,) so every function below can assume a simple
    # 1D list of labels.
    return arr.ravel() if arr.ndim > 1 else arr


# Receives: `y_true` and `y_pred` — the true and predicted labels, and an
#   optional `labels` list pinning down exactly which classes to use.
# Calculates: either returns the caller-supplied `labels` as-is, or — if
#   none was given — works out every distinct label that appears anywhere
#   in `y_true` or `y_pred`.
# Returns: a sorted (or caller-ordered) 1-D array of the class labels to
#   evaluate over.
def _resolve_labels(
    y_true: NDArray, y_pred: NDArray, labels: Optional[Union[Sequence, NDArray]] = None
) -> NDArray:
    """Sorted set of class labels to evaluate over, unless the caller pins one down."""
    # If the caller explicitly said "score these exact labels, in this exact
    # order", respect that instead of guessing.
    if labels is not None:
        return np.asarray(labels)
    # Otherwise, figure out the full set of classes ourselves: stack the
    # true and predicted labels into one array, then np.unique both removes
    # duplicates and sorts the result — e.g. y_true=[0,1,0], y_pred=[1,1,2]
    # gives [0, 1, 2].
    return np.unique(np.concatenate([y_true, y_pred]))


# Receives: `numerator` and `denominator` — two same-shaped arrays (or
#   scalars) to divide, and `fill_value` — what to report wherever the
#   denominator is exactly 0.
# Calculates: the elementwise division `numerator / denominator`, except
#   any position where `denominator` is 0 is replaced with `fill_value`
#   instead of raising a divide-by-zero warning or producing NaN/inf.
# Returns: an array (or array-like scalar result) the same shape as the
#   inputs.
def _safe_divide(numerator: NDArray, denominator: NDArray, fill_value: float) -> NDArray:
    """Elementwise division that substitutes `fill_value` instead of raising or
    producing NaN/inf wherever the denominator is zero (e.g. a class that
    was never predicted, so precision has nothing to divide by)."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    # We can't divide by 0 directly (numpy would print a runtime warning and
    # produce inf/NaN), so first build a "safe" version of the denominator
    # that swaps every 0 for a 1 — dividing by 1 is harmless and we'll throw
    # away that result anyway wherever the real denominator was actually 0.
    safe_denom = np.where(denominator == 0, 1.0, denominator)
    # np.where(condition, value_if_true, value_if_false), applied element by
    # element: wherever the ORIGINAL denominator was 0, use `fill_value`
    # instead of doing the division at all; everywhere else, do the normal
    # division using the safe denominator computed above.
    return np.where(denominator == 0, fill_value, numerator / safe_denom)


# Receives: `y_true` and `y_pred` — the true and predicted labels, and an
#   optional `labels` list controlling which classes (and in what order) to
#   include.
# Calculates: for every combination of (true label, predicted label), how
#   many samples fall into that combination.
# Returns: a square 2-D array of counts, shape (n_labels, n_labels), where
#   row = true class and column = predicted class.
def confusion_matrix(
    y_true: Union[Sequence, NDArray],
    y_pred: Union[Sequence, NDArray],
    labels: Optional[Union[Sequence, NDArray]] = None,
) -> NDArray:
    """Build a confusion matrix: rows are the true class, columns are the
    predicted class, so ``matrix[i, j]`` counts samples whose true label was
    ``labels[i]`` and whose predicted label was ``labels[j]``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)
    labels : array-like, optional
        Which labels to include and in what order. Defaults to every label
        that shows up in ``y_true`` or ``y_pred``, sorted. Samples whose
        true or predicted label isn't in this list are simply not counted
        anywhere, matching scikit-learn's behaviour.

    Returns
    -------
    NDArray of shape (n_labels, n_labels), dtype int64
    """
    y_true = _as_label_array(y_true)
    y_pred = _as_label_array(y_pred)
    check_same_length(y_true, y_pred, "classification")

    resolved_labels = _resolve_labels(y_true, y_pred, labels)
    n_labels = resolved_labels.shape[0]
    # A dict mapping each label value to its row/column position in the
    # matrix — e.g. {"cat": 0, "dog": 1, "bird": 2}. This lets us convert
    # arbitrary label values (strings, non-consecutive integers, ...) into
    # plain array indices.
    label_to_idx = {label: idx for idx, label in enumerate(resolved_labels)}

    # -1 marks a label outside the requested set; those rows/cols get dropped below.
    # `label_to_idx.get(v, -1)` looks up the position of value `v`, or
    # returns -1 if `v` isn't one of our known labels at all. We do this for
    # every entry of y_true and y_pred (np.fromiter builds an array from a
    # Python generator expression, which is what `(... for v in y_true)` is).
    row_idx = np.fromiter(
        (label_to_idx.get(v, -1) for v in y_true), dtype=np.int64, count=y_true.size
    )
    col_idx = np.fromiter(
        (label_to_idx.get(v, -1) for v in y_pred), dtype=np.int64, count=y_pred.size
    )
    # Boolean mask: True for samples where BOTH the true and predicted label
    # were recognised (i.e. neither index came back as -1).
    keep = (row_idx >= 0) & (col_idx >= 0)

    # Same "flatten a 2D (row, col) address into one 1D index" trick used
    # elsewhere in this library: flat_idx = row * n_columns + col. Only the
    # "kept" samples (those with valid labels) contribute.
    flat_idx = row_idx[keep] * n_labels + col_idx[keep]
    # Count how many samples land at each flat index, then fold the flat
    # list of counts back into an (n_labels, n_labels) grid — this is the
    # actual confusion matrix.
    counts = np.bincount(flat_idx, minlength=n_labels * n_labels)
    return counts.reshape(n_labels, n_labels).astype(np.int64)


# Receives: `y_true` and `y_pred` — the true and predicted labels.
# Calculates: what fraction of the two arrays match at the same position.
# Returns: a single float between 0 and 1.
def accuracy_score(y_true: Union[Sequence, NDArray], y_pred: Union[Sequence, NDArray]) -> float:
    """Fraction of predictions that exactly match the true label.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    float in [0, 1]
    """
    y_true = _as_label_array(y_true)
    y_pred = _as_label_array(y_pred)
    check_same_length(y_true, y_pred, "classification")
    # `y_true == y_pred` compares element by element and produces an array
    # of True/False values, one per sample. `np.mean` on a boolean array
    # treats True as 1 and False as 0, so the mean is exactly "fraction of
    # samples where the comparison was True" — i.e. the accuracy.
    return float(np.mean(y_true == y_pred))


# Receives: `y_true` and `y_pred` — the true and predicted labels, and
#   `labels` — which classes to compute counts for.
# Calculates: builds the confusion matrix, then reads off, per class, how
#   many true positives, false positives, and false negatives there are
#   (plus the "support" — how many samples truly belong to that class).
# Returns: four arrays, one entry per class: (true_positive, false_positive,
#   false_negative, support).
def _per_class_counts(
    y_true: NDArray, y_pred: NDArray, labels: NDArray
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """True/false positives, false negatives and support, one entry per label,
    read straight off the confusion matrix's diagonal and margins."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # np.diag(cm) reads off the diagonal entries of the matrix — for row/
    # column i, that's cm[i, i], the count of samples whose true class AND
    # predicted class were both label i. That's exactly the definition of a
    # "true positive" for that class.
    true_positive = np.diag(cm).astype(np.float64)
    # `cm.sum(axis=0)` adds up each COLUMN — i.e. "how many samples were
    # predicted as this class, regardless of what they truly were". Take
    # away the true positives and what's left is samples wrongly predicted
    # as this class: the false positives.
    false_positive = cm.sum(axis=0) - true_positive
    # `cm.sum(axis=1)` adds up each ROW — "how many samples truly belong to
    # this class, regardless of what was predicted". Subtracting true
    # positives leaves samples that truly belonged here but were predicted
    # as something else: the false negatives.
    false_negative = cm.sum(axis=1) - true_positive
    # "Support" is just how many samples truly belong to each class — the
    # same row totals used just above.
    support = cm.sum(axis=1)
    return true_positive, false_positive, false_negative, support


# Receives: `precision` and `recall` — arrays (or scalars) of per-class
#   precision/recall values, `beta` — how much more weight to give recall
#   over precision, and `zero_division` — the fallback value for a 0/0 case.
# Calculates: the general F-beta score formula, which becomes the familiar
#   F1 score when `beta=1.0`.
# Returns: an array (or scalar) of F-beta scores, same shape as the inputs.
def _f_beta(precision: NDArray, recall: NDArray, beta: float, zero_division: float) -> NDArray:
    # The general F-beta formula weights recall `beta` times as strongly as
    # precision. beta=1 (the default used by f1_score below) weights them
    # equally, which is the ordinary "harmonic mean of precision and recall".
    beta_sq = beta ** 2
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = beta_sq * precision + recall
    return _safe_divide(numerator, denominator, zero_division)


# Receives: `y_true`/`y_pred` — the true and predicted labels; `average` —
#   how to combine per-class scores into one number (or None for no
#   combining); `pos_label` — which class counts as "positive" for binary
#   scoring; `labels` — which classes to include; `beta` — the F-beta
#   weighting; `zero_division` — the fallback for undefined ratios.
# Calculates: precision, recall and F-beta score together (since they share
#   the same underlying true/false positive/negative counts), combined
#   according to whichever `average` mode was requested.
# Returns: a 4-tuple (precision, recall, f_score, support) — each entry is
#   either a single float/int (when averaging to one number) or an array
#   (when `average=None`, one value per class).
def precision_recall_fscore_support(
    y_true: Union[Sequence, NDArray],
    y_pred: Union[Sequence, NDArray],
    average: Optional[str] = "binary",
    pos_label: Union[int, str] = 1,
    labels: Optional[Union[Sequence, NDArray]] = None,
    beta: float = 1.0,
    zero_division: float = 0.0,
):
    """Precision, recall and F-beta score, computed together since they all
    come from the same per-class TP/FP/FN counts.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)
    average : {'binary', 'micro', 'macro', 'weighted', None}, default='binary'
        'binary' only makes sense for two classes and reports the metrics
        for ``pos_label``. 'micro' pools every TP/FP/FN across classes
        before dividing. 'macro' is the unweighted per-class mean.
        'weighted' weights each class's score by its support. ``None``
        returns the raw per-class arrays.
    pos_label : the label counted as "positive" when average='binary'.
    labels : which classes to score; defaults to every label seen in
        ``y_true`` or ``y_pred``.
    beta : weight of recall relative to precision in the F-score. 1.0 gives
        the standard F1.
    zero_division : value to report for a class where the denominator
        (e.g. no predicted positives at all) would otherwise be 0/0.

    Returns
    -------
    precision, recall, f_score : float, or NDArray when average=None
    support : int, or NDArray when average=None
    """
    y_true = _as_label_array(y_true)
    y_pred = _as_label_array(y_pred)
    check_same_length(y_true, y_pred, "classification")

    all_labels = _resolve_labels(y_true, y_pred, labels)

    if average == "binary":
        if labels is None and all_labels.size > 2:
            raise ValueError(
                f"average='binary' needs exactly 2 classes, found {all_labels.size} "
                f"({all_labels.tolist()}); use average='macro', 'micro' or 'weighted' instead"
            )
        # pos_label might not have shown up in this particular batch of
        # predictions at all — union it in so it still gets a (zero) row
        eval_labels = np.union1d(all_labels, [pos_label])
        tp, fp, fn, support = _per_class_counts(y_true, y_pred, eval_labels)
        # Find *where* in the eval_labels array pos_label ended up, so we
        # know which entry of tp/fp/fn/support belongs to the positive
        # class. np.flatnonzero returns the indices where a boolean array is
        # True; [0] takes the first (and only) match.
        idx = int(np.flatnonzero(eval_labels == pos_label)[0])
        # Precision = TP / (TP + FP): of everything we predicted positive,
        # what fraction really was?
        precision = _safe_divide(tp[idx], tp[idx] + fp[idx], zero_division)
        # Recall = TP / (TP + FN): of everything that really was positive,
        # what fraction did we catch?
        recall = _safe_divide(tp[idx], tp[idx] + fn[idx], zero_division)
        f_score = _f_beta(precision, recall, beta, zero_division)
        return float(precision), float(recall), float(f_score), int(support[idx])

    # For every averaging mode below, first get the per-class arrays (one
    # precision/recall/F-score value per label) — these are the building
    # blocks every average is computed from.
    tp, fp, fn, support = _per_class_counts(y_true, y_pred, all_labels)
    precision = _safe_divide(tp, tp + fp, zero_division)
    recall = _safe_divide(tp, tp + fn, zero_division)
    f_score = _f_beta(precision, recall, beta, zero_division)

    if average is None:
        # Caller wants the raw per-class arrays with no averaging at all —
        # used by classification_report to print one row per class.
        return precision, recall, f_score, support
    if average == "micro":
        # "Micro" averaging pools every class's TP/FP/FN into one grand
        # total FIRST, then computes a single precision/recall/F-score from
        # those totals — classes with more samples naturally count more.
        p = _safe_divide(tp.sum(), tp.sum() + fp.sum(), zero_division)
        r = _safe_divide(tp.sum(), tp.sum() + fn.sum(), zero_division)
        micro_f = _f_beta(p, r, beta, zero_division)
        return float(p), float(r), float(micro_f), int(support.sum())
    if average == "macro":
        # "Macro" averaging computes each class's metric SEPARATELY (as
        # above) and then takes the plain, unweighted average across
        # classes — a tiny class counts exactly as much as a huge one.
        return (
            float(precision.mean()),
            float(recall.mean()),
            float(f_score.mean()),
            int(support.sum()),
        )
    if average == "weighted":
        # "Weighted" averaging is like macro, but each class's contribution
        # to the average is scaled by how many samples it actually has
        # (its "support") — bigger classes count more, but each class's
        # own precision/recall is still computed independently first.
        total = support.sum()
        if total == 0:
            return float(zero_division), float(zero_division), float(zero_division), 0
        weights = support / total
        return (
            float(np.sum(precision * weights)),
            float(np.sum(recall * weights)),
            float(np.sum(f_score * weights)),
            int(total),
        )
    raise ValueError(
        f"unknown average {average!r}; expected 'binary', 'micro', 'macro', 'weighted' or None"
    )


# Receives: same inputs as `precision_recall_fscore_support` (y_true,
#   y_pred, average, pos_label, labels, zero_division).
# Calculates: delegates entirely to `precision_recall_fscore_support`.
# Returns: just the precision value/array from that function's result — of
#   everything predicted positive, the fraction that actually was.
def precision_score(
    y_true: Union[Sequence, NDArray],
    y_pred: Union[Sequence, NDArray],
    average: Optional[str] = "binary",
    pos_label: Union[int, str] = 1,
    labels: Optional[Union[Sequence, NDArray]] = None,
    zero_division: float = 0.0,
):
    """Of everything predicted positive, the fraction that actually was.
    See `precision_recall_fscore_support` for the parameters — this just
    returns the precision half of that result.
    """
    # Reuse the combined function above and just keep the first of the four
    # values it returns. The underscores are a Python convention meaning
    # "this returned value on purpose ignored, we don't need it here".
    precision, _, _, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average=average, pos_label=pos_label, labels=labels, zero_division=zero_division,
    )
    return precision


# Receives: same inputs as `precision_recall_fscore_support`.
# Calculates: delegates entirely to `precision_recall_fscore_support`.
# Returns: just the recall value/array from that function's result — of
#   everything that actually was positive, the fraction predicted positive.
def recall_score(
    y_true: Union[Sequence, NDArray],
    y_pred: Union[Sequence, NDArray],
    average: Optional[str] = "binary",
    pos_label: Union[int, str] = 1,
    labels: Optional[Union[Sequence, NDArray]] = None,
    zero_division: float = 0.0,
):
    """Of everything that actually was positive, the fraction predicted
    positive. See `precision_recall_fscore_support` for the parameters.
    """
    # Same idea as precision_score, but this time keep the second value
    # (recall) and discard the rest.
    _, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average=average, pos_label=pos_label, labels=labels, zero_division=zero_division,
    )
    return recall


# Receives: same inputs as `precision_recall_fscore_support`.
# Calculates: delegates entirely to `precision_recall_fscore_support` (with
#   `beta` fixed at its default of 1.0, i.e. the standard F1 definition).
# Returns: just the F1 value/array from that function's result — the
#   harmonic mean of precision and recall.
def f1_score(
    y_true: Union[Sequence, NDArray],
    y_pred: Union[Sequence, NDArray],
    average: Optional[str] = "binary",
    pos_label: Union[int, str] = 1,
    labels: Optional[Union[Sequence, NDArray]] = None,
    zero_division: float = 0.0,
):
    """Harmonic mean of precision and recall. See
    `precision_recall_fscore_support` for the parameters.
    """
    # Same idea again, keeping the third value (F1, since beta defaults to
    # 1.0 inside precision_recall_fscore_support).
    _, _, f_score, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average=average, pos_label=pos_label, labels=labels, zero_division=zero_division,
    )
    return f_score


# Receives: `y_true`/`y_pred` — the true and predicted labels; `labels` —
#   which classes to include; `target_names` — friendly display names for
#   those classes; `digits` — decimal places to print; `zero_division` —
#   fallback for undefined ratios.
# Calculates: precision/recall/F1/support for every class individually,
#   plus overall accuracy and the macro-average and weighted-average
#   summaries, then formats all of it into aligned table rows.
# Returns: a single multi-line string — the printable classification report.
def classification_report(
    y_true: Union[Sequence, NDArray],
    y_pred: Union[Sequence, NDArray],
    labels: Optional[Union[Sequence, NDArray]] = None,
    target_names: Optional[Sequence[str]] = None,
    digits: int = 2,
    zero_division: float = 0.0,
) -> str:
    """Print-ready summary table: precision/recall/F1/support per class,
    plus overall accuracy and macro/weighted averages — the table you'd
    paste straight into a notebook cell or a report.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)
    labels : which classes to include; defaults to every label seen.
    target_names : display names for each label, in the same order as
        ``labels`` (or the default sorted label order if ``labels`` is
        omitted).
    digits : how many decimal places to print.
    zero_division : value used when a class has an undefined precision or
        recall (see `precision_recall_fscore_support`).

    Returns
    -------
    str
    """
    y_true = _as_label_array(y_true)
    y_pred = _as_label_array(y_pred)
    check_same_length(y_true, y_pred, "classification")

    all_labels = _resolve_labels(y_true, y_pred, labels)
    if target_names is not None:
        # If the caller gave us friendly display names ("cat", "dog", ...)
        # instead of raw label values (0, 1, ...), they need to line up
        # one-to-one with the labels we're about to report on.
        if len(target_names) != len(all_labels):
            raise ValueError(
                f"target_names has {len(target_names)} entries "
                f"but there are {len(all_labels)} labels"
            )
        names = [str(name) for name in target_names]
    else:
        # No friendly names given — just print the raw label values
        # themselves (turned into strings so formatting works uniformly).
        names = [str(label) for label in all_labels]

    # `average=None` asks for the raw per-class arrays (one number per
    # label), which is exactly what we need to print one row per class.
    precision, recall, f_score, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=all_labels, zero_division=zero_division
    )
    accuracy = accuracy_score(y_true, y_pred)
    # Also compute the two summary rows shown at the bottom of the table.
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=all_labels, zero_division=zero_division
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=all_labels, zero_division=zero_division
    )

    # Figure out how wide the left-hand "class name" column needs to be so
    # every row (including the longest label, "weighted avg") lines up.
    name_width = max(len(name) for name in names + ["weighted avg"])
    # `{'':>{name_width}}` prints an empty string right-aligned in a column
    # `name_width` characters wide (i.e. just blank padding), and
    # `{'precision':>9}` prints the word "precision" right-aligned in a
    # 9-character column — this builds the column header row.
    header = f"{'':>{name_width}}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>9}"
    total_support = int(support.sum())

    rows = [header, ""]
    # zip(...) walks through all five per-class lists (names, precision,
    # recall, f_score, support) in lockstep, one class at a time, so each
    # loop iteration has everything needed to print that class's row.
    for name, p, r, f, s in zip(names, precision, recall, f_score, support):
        rows.append(
            f"{name:>{name_width}}  {p:>9.{digits}f}  {r:>9.{digits}f}  "
            f"{f:>9.{digits}f}  {int(s):>9d}"
        )
    rows.append("")  # a blank line separates the per-class rows from the summary rows
    rows.append(
        f"{'accuracy':>{name_width}}  {'':>9}  {'':>9}  "
        f"{accuracy:>9.{digits}f}  {total_support:>9d}"
    )
    rows.append(
        f"{'macro avg':>{name_width}}  {p_macro:>9.{digits}f}  {r_macro:>9.{digits}f}  "
        f"{f_macro:>9.{digits}f}  {total_support:>9d}"
    )
    rows.append(
        f"{'weighted avg':>{name_width}}  {p_weighted:>9.{digits}f}  {r_weighted:>9.{digits}f}  "
        f"{f_weighted:>9.{digits}f}  {total_support:>9d}"
    )
    # Join every row together with newlines into one big printable string.
    return "\n".join(rows)


# Receives: `y_true` — the true binary labels, `y_score` — predicted scores/
#   probabilities, and `pos_label` — which label counts as "positive".
# Calculates: confirms shapes match, confirms there are exactly two classes
#   present, and counts how many positive vs. negative samples there are.
# Returns: a 3-tuple — a boolean mask marking the positive-class samples,
#   the count of positives, and the count of negatives.
def _validate_binary_scores(
    y_true: NDArray, y_score: NDArray, pos_label: Union[int, str]
) -> Tuple[NDArray, int, int]:
    """Shared setup for the two probability-based metrics below: check shapes,
    confirm this is really a two-class problem, and split out how many
    positives/negatives there are to score against."""
    check_same_length(y_true, y_score, "classification")
    labels = np.unique(y_true)
    # ROC-AUC and average precision are only defined for exactly two
    # classes (there's no single "curve" for 3+ classes without extra
    # machinery this library doesn't implement), so reject anything else
    # up front with a clear error rather than producing a nonsense number.
    if labels.size != 2:
        raise ValueError(
            f"this metric is defined for binary classification only, found {labels.size} "
            f"classes ({labels.tolist()})"
        )
    # Boolean mask marking which samples are the "positive" class.
    is_positive = y_true == pos_label
    n_pos = int(is_positive.sum())
    n_neg = y_true.size - n_pos
    # If every sample is the same class, there's no "randomly chosen
    # positive vs. randomly chosen negative" comparison to make at all.
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need at least one sample of each class to score this metric")
    return is_positive, n_pos, n_neg


# Receives: `y_true` — the true labels, `y_score` — a 2-D probability
#   matrix (one column per class), and `labels` — the caller-supplied class
#   order for those columns, or None.
# Calculates: works out which class each column of `y_score` belongs to
#   (every distinct label in `y_true`, sorted, unless `labels` pins down an
#   explicit order), then confirms that count matches the number of columns
#   actually supplied.
# Returns: a 1-D array of class labels, one per column of `y_score`, in
#   column order.
def _resolve_multiclass_labels(
    y_true: NDArray, y_score: NDArray, labels: Optional[Union[Sequence, NDArray]]
) -> NDArray:
    """Class labels matching the columns of a multiclass `y_score` matrix,
    checked for a matching column count."""
    # No explicit column order given — fall back to every label seen in
    # y_true, sorted, which is the same order `predict_proba` implementations
    # in this library use for their `classes_` attribute.
    resolved = np.unique(y_true) if labels is None else np.asarray(labels)
    if resolved.size != y_score.shape[1]:
        raise ValueError(
            f"y_score has {y_score.shape[1]} columns but {resolved.size} labels "
            f"were resolved ({resolved.tolist()}); pass `labels=` explicitly if "
            "the column order doesn't match the sorted class order"
        )
    return resolved


# Receives: `y_true` — the true labels, `y_score` — a 2-D probability
#   matrix (one column per class), `labels` — the class each column
#   represents, `average` — how to combine the per-class scores, and
#   `binary_metric` — a function that scores one class-vs-rest column
#   (either the ROC-AUC or the average-precision rank-sum logic below).
# Calculates: One-vs-Rest scoring — treats each class in turn as "positive"
#   against every other class as "negative", scores that class's own
#   probability column with `binary_metric`, then combines every class's
#   score into a single number.
# Returns: a single float — the macro (unweighted) or support-weighted
#   average of the per-class scores.
def _multiclass_ovr_average(
    y_true: NDArray,
    y_score: NDArray,
    labels: NDArray,
    average: str,
    binary_metric,
) -> float:
    """Combine per-class One-vs-Rest scores into one macro or weighted average."""
    n_classes = labels.shape[0]
    per_class_scores = np.empty(n_classes)
    support = np.empty(n_classes)
    # One pass per class: collapse the multiclass y_true down to a plain
    # "is this sample really this class?" boolean, so the binary metric
    # sees exactly two classes (this one vs. every other class lumped
    # together), scored against that class's own probability column.
    for idx, label in enumerate(labels):
        is_label = y_true == label
        per_class_scores[idx] = binary_metric(is_label, y_score[:, idx], True)
        support[idx] = np.sum(is_label)

    if average == "macro":
        # Every class counts equally, regardless of how many samples it has.
        return float(per_class_scores.mean())
    if average == "weighted":
        # Classes with more samples pull the average further toward their score.
        total = support.sum()
        return float(np.sum(per_class_scores * support) / total)
    raise ValueError(f"unknown average {average!r}; expected 'macro' or 'weighted'")


# Receives: `y_true` — the true binary labels, `y_score` — predicted
#   probabilities/scores for the positive class, and `pos_label` — which
#   label counts as "positive".
# Calculates: ranks every predicted score from lowest to highest, then works
#   out (via a rank-sum formula) the probability that a randomly chosen
#   positive sample outranks a randomly chosen negative sample.
# Returns: a single float between 0 and 1 — the ROC-AUC score.
def _binary_roc_auc_score(
    y_true: NDArray, y_score: NDArray, pos_label: Union[int, str]
) -> float:
    """The two-class ROC-AUC computation, shared by the public `roc_auc_score`
    entry point (binary case) and the multiclass One-vs-Rest average above."""
    is_positive, n_pos, n_neg = _validate_binary_scores(y_true, y_score, pos_label)

    # Rank every predicted score from lowest (rank 1) to highest (see
    # `_shared.rank` for exactly how ties are handled). A positive sample
    # that scored higher than most other samples will end up with a large
    # rank number.
    score_ranks = rank(y_score)
    # Add up the ranks belonging ONLY to the positive-class samples
    # (boolean-mask indexing again: `score_ranks[is_positive]` keeps just
    # those entries).
    sum_ranks_positive = score_ranks[is_positive].sum()
    # This is the standard Mann-Whitney U formula in disguise: if the ranks
    # of the positive samples were as low as possible (i.e. all packed at
    # the bottom), sum_ranks_positive would equal n_pos*(n_pos+1)/2 exactly,
    # giving an AUC of 0. If they were as high as possible (packed at the
    # top, meaning every positive outranks every negative), the fraction
    # works out to exactly 1. Values in between measure, proportionally, how
    # often a positive sample outranks a negative one.
    return float((sum_ranks_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# Receives: `y_true` — the true labels, `y_score` — either a 1-D array of
#   positive-class scores (binary) or a 2-D probability matrix, one column
#   per class (multiclass); `pos_label` — which label counts as "positive"
#   in the binary case; `average` — how to combine per-class scores in the
#   multiclass case; `labels` — the class each column of a multiclass
#   `y_score` represents.
# Calculates: the binary ROC-AUC directly, or — when `y_score` has more than
#   one column — the One-vs-Rest ROC-AUC for every class, combined by
#   `average`.
# Returns: a single float between 0 and 1 — the ROC-AUC score.
def roc_auc_score(
    y_true: Union[Sequence, NDArray],
    y_score: Union[Sequence, NDArray],
    pos_label: Union[int, str] = 1,
    average: str = "macro",
    labels: Optional[Union[Sequence, NDArray]] = None,
) -> float:
    """Area under the ROC curve: the probability that a randomly chosen
    positive sample gets a higher predicted score than a randomly chosen
    negative one. 1.0 is a perfect ranking, 0.5 is no better than a coin
    flip, and unlike accuracy/F1 it doesn't depend on picking a decision
    threshold at all — it judges the whole ranking `predict_proba` produces.

    Computed as a rank-sum (Mann-Whitney U) statistic rather than by
    building the ROC curve and integrating under it — mathematically the
    same number, far less code.

    For more than two classes, pass the full probability matrix as
    `y_score` (shape ``(n_samples, n_classes)``, e.g. `model.predict_proba(X)`)
    and this scores it One-vs-Rest: every class takes a turn as "positive"
    against all the others, and the per-class scores are combined with
    `average`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities or decision-function scores for the
        positive class (binary), or one column per class (multiclass).
    pos_label : the label treated as the positive class. Only used in the
        binary case (1-D `y_score`).
    average : {'macro', 'weighted'}, default='macro'
        How to combine per-class scores in the multiclass case. Ignored for
        binary `y_score`.
    labels : which class each column of a multiclass `y_score` represents,
        in column order. Defaults to every label seen in `y_true`, sorted.

    Returns
    -------
    float in [0, 1]
    """
    y_true = _as_label_array(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_score.ndim == 2 and y_score.shape[1] > 1:
        resolved_labels = _resolve_multiclass_labels(y_true, y_score, labels)
        return _multiclass_ovr_average(
            y_true, y_score, resolved_labels, average, _binary_roc_auc_score
        )
    # A single-column 2-D array (shape (n, 1)) is just the binary case
    # written awkwardly — flatten it down to 1-D before scoring.
    if y_score.ndim == 2:
        y_score = y_score.ravel()
    return _binary_roc_auc_score(y_true, y_score, pos_label)


# Receives: `y_true` — the true binary labels, `y_score` — predicted
#   probabilities/scores for the positive class, and `pos_label` — which
#   label counts as "positive".
# Calculates: walks the samples from the highest predicted score down to
#   the lowest, tracking precision and recall at every distinct score
#   threshold, and sums up precision weighted by the recall gained at each
#   step.
# Returns: a single float between 0 and 1 — the area under the (step-shaped,
#   uninterpolated) precision-recall curve.
def _binary_average_precision_score(
    y_true: NDArray, y_score: NDArray, pos_label: Union[int, str]
) -> float:
    """The two-class average-precision computation, shared by the public
    `average_precision_score` entry point (binary case) and the multiclass
    One-vs-Rest average above."""
    is_positive, n_pos, _ = _validate_binary_scores(y_true, y_score, pos_label)

    # walk highest-score-first; ties keep their input order, same convention sklearn uses.
    # `np.argsort(-y_score)` sorts by the NEGATED scores in ascending order,
    # which is the same as sorting the original scores in DESCENDING order
    # (highest predicted confidence first). We're imagining a classifier
    # that starts by only calling the single most-confident sample
    # "positive", then gradually lowers its threshold to include more and
    # more samples.
    order = np.argsort(-y_score, kind="mergesort")
    is_positive_sorted = is_positive[order].astype(np.float64)
    scores_sorted = y_score[order]

    # collapse runs of equal scores down to one threshold each — precision/recall
    # only need to be evaluated once per distinct score, at the run's last index.
    # np.diff(scores_sorted) computes each element minus the one before it;
    # wherever that's nonzero, the score changed, marking a new "threshold".
    # `np.where(...)[0]` turns that boolean array into the actual positions
    # where a change happened.
    run_boundary = np.where(np.diff(scores_sorted) != 0)[0]
    # np.r_[a, b] concatenates arrays end-to-end. We always include the very
    # last position too (every run ends there even if the score never
    # changes again), since that's still a valid threshold to evaluate.
    threshold_idx = np.r_[run_boundary, len(scores_sorted) - 1]

    # np.cumsum computes a "running total": cumsum([1,0,1,1])=[1,1,2,3]. So
    # np.cumsum(is_positive_sorted) at any position tells us "how many
    # positives have we seen so far, reading top-to-bottom from the highest
    # score down to here". Indexing that running total at our threshold
    # positions gives the true-positive count AT each threshold.
    true_positives = np.cumsum(is_positive_sorted)[threshold_idx]
    # At threshold position i (0-based), we've looked at i+1 samples total
    # (everyone from the top down to and including this one) and called all
    # of them "positive" — that's how many predicted-positives there are so
    # far.
    predicted_positives = threshold_idx + 1
    # Precision at this threshold: of everyone called positive so far, what
    # fraction really were? Recall at this threshold: of all the true
    # positives that exist anywhere in the dataset, what fraction have we
    # found so far?
    precision = true_positives / predicted_positives
    recall = true_positives / n_pos

    # prepend the (recall=0, precision=1) starting point so the first real
    # threshold's recall *gain* is measured from zero. This matches the
    # convention that "before we've called anything positive, recall is 0
    # and precision is trivially perfect".
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    # np.diff(recall) gives the recall GAINED at each step (how much higher
    # recall got compared to the previous threshold). Multiplying each gain
    # by the precision achieved at that step, and summing everything up, is
    # exactly the area under the (step-shaped, uninterpolated)
    # precision-recall curve.
    return float(np.sum(np.diff(recall) * precision[1:]))


# Receives: `y_true` — the true labels, `y_score` — either a 1-D array of
#   positive-class scores (binary) or a 2-D probability matrix, one column
#   per class (multiclass); `pos_label` — which label counts as "positive"
#   in the binary case; `average` — how to combine per-class scores in the
#   multiclass case; `labels` — the class each column of a multiclass
#   `y_score` represents.
# Calculates: the binary average precision directly, or — when `y_score`
#   has more than one column — the One-vs-Rest average precision for every
#   class, combined by `average`.
# Returns: a single float between 0 and 1 — the area under the (step-shaped,
#   uninterpolated) precision-recall curve.
def average_precision_score(
    y_true: Union[Sequence, NDArray],
    y_score: Union[Sequence, NDArray],
    pos_label: Union[int, str] = 1,
    average: str = "macro",
    labels: Optional[Union[Sequence, NDArray]] = None,
) -> float:
    """Area under the precision-recall curve, summarised without any
    interpolation: walk the samples from highest predicted score to lowest,
    and at every point where the score changes, weight that point's
    precision by how much recall was gained getting there. Preferred over
    ROC-AUC when positives are rare, since it doesn't give credit for the
    (usually huge) pool of easy true negatives the way ROC-AUC's false
    positive rate does.

    For more than two classes, pass the full probability matrix as
    `y_score` (shape ``(n_samples, n_classes)``, e.g. `model.predict_proba(X)`)
    and this scores it One-vs-Rest: every class takes a turn as "positive"
    against all the others, and the per-class scores are combined with
    `average`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities or decision-function scores for the
        positive class (binary), or one column per class (multiclass).
    pos_label : the label treated as the positive class. Only used in the
        binary case (1-D `y_score`).
    average : {'macro', 'weighted'}, default='macro'
        How to combine per-class scores in the multiclass case. Ignored for
        binary `y_score`.
    labels : which class each column of a multiclass `y_score` represents,
        in column order. Defaults to every label seen in `y_true`, sorted.

    Returns
    -------
    float in [0, 1]
    """
    y_true = _as_label_array(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_score.ndim == 2 and y_score.shape[1] > 1:
        resolved_labels = _resolve_multiclass_labels(y_true, y_score, labels)
        return _multiclass_ovr_average(
            y_true, y_score, resolved_labels, average, _binary_average_precision_score
        )
    # A single-column 2-D array (shape (n, 1)) is just the binary case
    # written awkwardly — flatten it down to 1-D before scoring.
    if y_score.ndim == 2:
        y_score = y_score.ravel()
    return _binary_average_precision_score(y_true, y_score, pos_label)
