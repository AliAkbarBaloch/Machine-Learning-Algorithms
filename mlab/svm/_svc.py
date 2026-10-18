"""Kernel SVM classifier (SVC) via Sequential Minimal Optimisation (SMO)."""

from itertools import combinations
from typing import NamedTuple, Optional, Union

import numpy as np

from mlab.svm._svm_utils import (
    KernelParams,
    compute_kernel,
    impute_column_medians,
    prepare_fit_inputs,
    resolve_gamma,
)


# A NamedTuple is an immutable, named-field tuple — no methods needed here,
# it just bundles the handful of numbers one `_BinarySVC` needs into a
# single object, since `SVC` may build several of these (one per class
# pair in One-vs-One mode) and passing them around as one config is
# simpler than seven loose arguments.
class _BinaryConfig(NamedTuple):
    """Immutable hyperparameter bundle for a binary SMO classifier."""

    # 'linear', 'rbf', or 'poly' — which kernel function this classifier
    # uses.
    kernel: str
    # Kernel coefficient (already resolved to a concrete number by the
    # time this reaches a `_BinarySVC`, never the 'scale'/'auto' strings).
    gamma: float
    # Degree of the polynomial kernel.
    degree: int
    # Independent term added inside the polynomial kernel.
    coef0: float
    # Regularisation strength C — the upper bound on every alpha.
    reg_param: float
    # How much KKT-condition slack is tolerated before SMO calls a sample
    # "not yet converged".
    tol: float
    # Maximum number of SMO outer-loop passes before giving up.
    max_iter: int


# Another plain NamedTuple: everything one `_BinarySVC.fit()` call learns,
# bundled into a single immutable object instead of several separate
# instance attributes.
class _SVFit(NamedTuple):
    """Immutable container for the fitted support-vector data of one binary classifier."""

    # The dual coefficient (Lagrange multiplier) for every support vector.
    alpha: np.ndarray
    # The {-1, +1} label of every support vector.
    sv_y: np.ndarray
    # The feature row of every support vector.
    sv_x: np.ndarray
    # The training-set row index of every support vector.
    sv_idx: np.ndarray
    # The fitted bias term b.
    bias: float
    # alpha * y for each support vector, pre-multiplied and reshaped to
    # (1, n_sv) — exactly what `decision_function` needs for its matrix
    # multiply against a kernel row.
    dual_coef: np.ndarray


class _SMOState:
    """Mutable optimisation state for a single SMO training run.

    Centralises ``alpha``, ``bias``, ``errors``, ``y_bin``, and
    ``kernel_mat`` so each inner function receives one argument instead of
    five, keeping argument lists within pylint limits.

    Optimization techniques
    -----------------------
    kernel caching / kernel matrix decomposition: The full Gram matrix is
        precomputed once and stored here; all pair updates read cached rows
        (O(n)) rather than recomputing the kernel (O(n²)).
    memory management: ``kernel_mat`` is stored as a C-contiguous float64
        array so row-slices stay cache-friendly throughout the loop.
        ``buf`` is a pre-allocated scratch array reused in the hot-path
        error-update step of ``_try_pair`` to avoid repeated heap allocation.
    numerical stability: ``recompute_errors`` recalculates the error vector
        from scratch every 50 iterations to prevent floating-point drift.
    adaptive algorithms: ``nb`` caches the non-bound alpha mask
        (``0 < alpha < C``); refreshed once per outer SMO step via
        ``refresh_nb`` to avoid O(n) recomputation per KKT violator.
    """

    # Receives: y_bin — the {-1, +1} training labels, and kernel_mat — the
    #   precomputed full Gram matrix for the training set.
    # Calculates: nothing yet — just initialises every piece of mutable
    #   state SMO needs to track as it iterates: the dual coefficients
    #   `alpha` (all starting at 0, meaning no sample is a support vector
    #   yet), the bias, the per-sample prediction error (starting as -y,
    #   since alpha=0 means every prediction is 0), and some scratch
    #   space/caches used to speed up the hot loop.
    # Returns: nothing — __init__ only sets up the object's fields.
    def __init__(self, y_bin: np.ndarray, kernel_mat: np.ndarray) -> None:
        self.alpha: np.ndarray = np.zeros(len(y_bin))
        self.bias: float = 0.0
        self.errors: np.ndarray = -y_bin.astype(float)
        self.y_bin: np.ndarray = y_bin
        self.kernel_mat: np.ndarray = kernel_mat
        self.buf: np.ndarray = np.empty(len(y_bin))
        self.nb: np.ndarray = np.zeros(len(y_bin), dtype=bool)

    # Receives: nothing besides self.
    # Calculates: the full error vector (predicted score minus true label,
    #   for every sample) directly from the current alpha/bias, instead of
    #   from the incrementally-updated `errors` array.
    # Returns: nothing — overwrites `self.errors` in place.
    def recompute_errors(self) -> None:
        """Recompute the error vector from scratch (numerical stability)."""
        self.errors = (
            self.kernel_mat @ (self.alpha * self.y_bin) + self.bias - self.y_bin
        )

    # Receives: tol — how much slack to allow before calling a sample a
    #   genuine KKT violation, and reg_param — the box constraint's upper
    #   bound C.
    # Calculates: which samples currently violate the SVM's
    #   Karush-Kuhn-Tucker optimality conditions — i.e. which ones the
    #   current alpha/bias values haven't yet correctly classified with
    #   enough margin, given their own alpha's position in [0, C].
    # Returns: an array of integer indices — the samples that still need
    #   updating.
    def kkt_violations(self, tol: float, reg_param: float) -> np.ndarray:
        """Return indices of KKT-violating samples (shrinking / convergence checking)."""
        rho = self.y_bin * self.errors
        mask = ((rho < -tol) & (self.alpha < reg_param)) | (
            (rho > tol) & (self.alpha > 0)
        )
        return np.where(mask)[0]

    # Receives: cap — the box constraint's upper bound C.
    # Calculates: which samples currently have a "non-bound" alpha (strictly
    #   between 0 and C, i.e. not pinned to either boundary) — these are
    #   the preferred candidates for the next pair update, since they're
    #   the ones the algorithm is still actively adjusting.
    # Returns: nothing — updates `self.nb` (the cached boolean mask) in
    #   place.
    def refresh_nb(self, cap: float) -> None:
        """Refresh the cached non-bound alpha mask (adaptive algorithms, memory management)."""
        np.greater(self.alpha, 0.0, out=self.nb)
        self.nb &= self.alpha < cap


# A NamedTuple, no methods — just a small return-value bundle so
# `_fit_binary_pair`/`_fit_ovo` can hand back "the classifiers I trained"
# and "which class pair each one belongs to" as one object instead of two
# separate return values that could get mismatched.
class _ClassifierBatch(NamedTuple):
    """Pairs of trained binary classifiers and their class-pair labels."""

    # Every `_BinarySVC` that was trained (one for binary problems, one
    # per class-pair for One-vs-One multiclass).
    classifiers: list
    # The (negative_class, positive_class) label pair each classifier in
    # `classifiers` was trained on, in the same order.
    pairs: list


# Another plain NamedTuple: bundles the standardised training data
# together with the exact mean/scale used to produce it, so both travel
# together as one value instead of three separately-passed arguments.
class _ScaledData(NamedTuple):
    """Training data bundled with fit-time StandardScaler parameters."""

    # The training feature matrix, already standardised to zero mean/unit
    # variance.
    x_scaled: np.ndarray
    # The per-feature mean subtracted to produce `x_scaled`.
    scaler_mean: np.ndarray
    # The per-feature standard deviation divided out to produce
    # `x_scaled`.
    scaler_scale: np.ndarray


# A NamedTuple bundling everything `SVC.fit()` learns and needs later —
# every property on the public `SVC` class (support_vectors_, dual_coef_,
# etc.) just reads straight out of one of these fields.
class _SVCFitState(NamedTuple):
    """Immutable container for the fully fitted state of an ``SVC`` instance."""

    # Every trained binary classifier (empty if the model was never
    # actually fit on more than one class, or on no data at all).
    classifiers: list
    # The (negative_class, positive_class) pair each classifier above was
    # trained on.
    pairs: list
    # The resolved (already-a-number) gamma value actually used for
    # training.
    gamma_val: float
    # If only a single class was ever seen during fit(), that one class
    # label — predict() then shortcuts straight to always returning it.
    # None otherwise.
    single_class: object
    # Every support vector across every trained binary classifier,
    # deduplicated.
    support_vectors: np.ndarray
    # The training-set row index of every one of those support vectors.
    support: np.ndarray
    # Every classifier's own alpha*y dual coefficients, flattened and
    # concatenated into one combined (1, n_total_sv) row.
    dual_coef: np.ndarray
    # One bias term per binary classifier.
    intercept: np.ndarray
    # An approximate support-vector count per class.
    n_support: np.ndarray
    # Every distinct class label seen during fit().
    classes: np.ndarray
    # Per-feature mean used to standardise the training data; None in the
    # degenerate empty-dataset case.
    scaler_mean: Optional[np.ndarray]
    # Per-feature standard deviation used for that same standardisation.
    scaler_scale: Optional[np.ndarray]


class _BinarySVC:
    """Internal binary SVC solved via Platt's Sequential Minimal Optimisation (SMO).

    Solves the dual QP:
        max  sum_i alpha_i - (1/2) sum_{i,j} alpha_i alpha_j y_i y_j K(xi, xj)
        s.t. 0 <= alpha_i <= C,   sum_i alpha_i y_i = 0

    Prediction:  f(x) = sum_{sv} alpha_i y_i K(x_i, x) + b

    Optimization notes
    ------------------
    sequential minimal optimization: Decomposes the QP into a series of
        two-variable sub-problems solved analytically (alpha selection,
        kkt conditions, convergence checking).
    precision handling: Scalar alpha clipping uses Python ``min``/``max``
        rather than ``np.clip`` to avoid NumPy dispatch overhead on scalars.
    adaptive algorithms: Level-1 heuristic selects j = argmax|Ej-Ei|; falls
        back to non-bound then all-sample seeded-random sweeps.
    """

    # Receives: config — the bundled kernel/regularisation/tolerance
    #   hyperparameters, and random_state — a seed (currently unused for
    #   randomness inside this class, but accepted for API symmetry).
    # Calculates: nothing yet — remembers the config and prepares an empty
    #   "not fitted" state.
    # Returns: nothing — __init__ only prepares the object.
    def __init__(
        self, config: _BinaryConfig, random_state: Optional[int] = None
    ) -> None:
        self._cfg: _BinaryConfig = config
        self._rng: np.random.Generator = np.random.default_rng(random_state)
        self._sv: Optional[_SVFit] = None

    # ------------------------------------------------------------------
    # Public fitted-attribute properties (set in fit, read-only outside)
    # ------------------------------------------------------------------

    # Receives: nothing besides self.
    # Calculates: nothing — reads the dual coefficients out of the fitted
    #   state.
    # Returns: the alpha value for every support vector, or empty if not
    #   yet fitted.
    @property
    def alpha_(self) -> np.ndarray:
        """Dual coefficients for the support vectors."""
        return self._sv.alpha if self._sv else np.empty(0)

    # Receives: nothing besides self.
    # Calculates: nothing — reads the support vectors' own labels.
    # Returns: the {-1, +1} label of every support vector, or empty if not
    #   yet fitted.
    @property
    def sv_y_(self) -> np.ndarray:
        """Class labels (+/-1) of the support vectors."""
        return self._sv.sv_y if self._sv else np.empty(0)

    # Receives: nothing besides self.
    # Calculates: nothing — reads the support vectors' own feature rows.
    # Returns: the feature matrix of every support vector, or an empty
    #   (0, 0) array if not yet fitted.
    @property
    def sv_x_(self) -> np.ndarray:
        """Feature matrix of the support vectors."""
        return self._sv.sv_x if self._sv else np.empty((0, 0))

    # Receives: nothing besides self.
    # Calculates: nothing — reads the support vectors' training-set row
    #   indices.
    # Returns: the original row index of every support vector, or empty if
    #   not yet fitted.
    @property
    def sv_idx_(self) -> np.ndarray:
        """Training indices of the support vectors."""
        return self._sv.sv_idx if self._sv else np.empty(0, dtype=int)

    # Receives: nothing besides self.
    # Calculates: nothing — reads the fitted bias term.
    # Returns: the bias term b, or 0.0 if not yet fitted.
    @property
    def b_(self) -> float:
        """Bias term."""
        return self._sv.bias if self._sv else 0.0

    # Receives: nothing besides self.
    # Calculates: nothing — reads the pre-multiplied alpha*y dual
    #   coefficients.
    # Returns: a (1, n_sv) row of alpha_i * y_i values, one per support
    #   vector — this is exactly what's needed to compute predictions via a
    #   single matrix multiply against a kernel matrix.
    @property
    def dual_coef_(self) -> np.ndarray:
        """Row vector of alpha * y for each support vector."""
        return self._sv.dual_coef if self._sv else np.empty((1, 0))

    # ------------------------------------------------------------------
    # Internal kernel helper
    # ------------------------------------------------------------------

    # Receives: x_arr1 and x_arr2 — two sets of samples.
    # Calculates: nothing new — just forwards to the shared
    #   `compute_kernel` helper with this classifier's own kernel type and
    #   hyperparameters.
    # Returns: the kernel (Gram) matrix between x_arr1 and x_arr2.
    def _compute_kernel(self, x_arr1: np.ndarray, x_arr2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between two sample arrays."""
        return compute_kernel(
            x_arr1, x_arr2, self._cfg.kernel,
            KernelParams(self._cfg.gamma, self._cfg.degree, self._cfg.coef0),
        )

    # ------------------------------------------------------------------
    # SMO pair-update helpers
    # ------------------------------------------------------------------

    # Receives: idx_i and idx_j — the two sample indices SMO wants to
    #   jointly update this step, and state — the current optimisation
    #   state (alpha, bias, errors, etc.).
    # Calculates: the SMO closed-form analytic update for exactly these two
    #   alphas: works out the valid box-constraint range for alpha_j,
    #   computes the unconstrained optimum via the two samples' kernel
    #   values and current errors, clips it into range, derives the
    #   matching alpha_i update (since the sum-to-zero constraint ties them
    #   together), and finally re-derives the bias.
    # Returns: True if this pair update actually changed anything by more
    #   than a tiny numerical threshold (and, in that case, mutates `state`
    #   in place with the new alpha/bias/errors); False if the pair
    #   couldn't be usefully updated (and `state` is left as-is).
    def _try_pair(self, idx_i: int, idx_j: int, state: _SMOState) -> bool:
        """Attempt one SMO pair update; mutates state in-place.

        Returns
        -------
        bool
            True when the pair was updated by more than the numerical threshold.
        """
        err_i = state.errors[idx_i]
        err_j = state.errors[idx_j]
        ai_old = float(state.alpha[idx_i])
        aj_old = float(state.alpha[idx_j])

        # The valid range for the new alpha_j depends on whether the two
        # samples share a class label or not — this comes directly from
        # the SVM dual's linear constraint (sum of alpha_i * y_i = 0)
        # combined with the box constraint 0 <= alpha <= C.
        if state.y_bin[idx_i] != state.y_bin[idx_j]:
            lower = max(0.0, state.alpha[idx_j] - state.alpha[idx_i])
            upper = min(
                self._cfg.reg_param,
                self._cfg.reg_param + state.alpha[idx_j] - state.alpha[idx_i],
            )
        else:
            lower = max(0.0, state.alpha[idx_i] + state.alpha[idx_j] - self._cfg.reg_param)
            upper = min(self._cfg.reg_param, state.alpha[idx_i] + state.alpha[idx_j])

        if lower >= upper:
            # No room to move alpha_j at all — this pair can't be updated.
            return False

        # `eta` is (twice the kernel cross-term) minus (both samples' own
        # self-kernel values) — this is the second derivative of the dual
        # objective with respect to alpha_j along this pair's constrained
        # direction. It must be strictly negative for the quadratic to have
        # a genuine maximum (rather than being flat or upward-curving).
        eta = (
            2.0 * state.kernel_mat[idx_i, idx_j]
            - state.kernel_mat[idx_i, idx_i]
            - state.kernel_mat[idx_j, idx_j]
        )
        if eta >= 0.0:
            return False

        # The unconstrained analytic optimum for alpha_j (Platt's SMO
        # formula), then immediately clipped into the [lower, upper] range
        # computed above.
        state.alpha[idx_j] -= state.y_bin[idx_j] * (err_i - err_j) / eta
        # Scalar clipping via Python min/max — avoids NumPy dispatch on scalars.
        state.alpha[idx_j] = min(max(float(state.alpha[idx_j]), lower), upper)
        if abs(state.alpha[idx_j] - aj_old) < 1e-5:
            # The update barely moved anything — not worth committing (and
            # floating-point noise could otherwise accumulate over many
            # near-zero "updates").
            state.alpha[idx_j] = aj_old
            return False

        # The dual's linear constraint ties alpha_i's change to alpha_j's
        # change exactly (in opposite proportion when labels differ, same
        # proportion when they match) — this restores that constraint.
        state.alpha[idx_i] += (
            state.y_bin[idx_i] * state.y_bin[idx_j] * (aj_old - state.alpha[idx_j])
        )

        # Two candidate new bias values, each derived from making one of
        # the two samples' own KKT condition exactly satisfied.
        b1 = (
            state.bias - err_i
            - state.y_bin[idx_i] * (state.alpha[idx_i] - ai_old) * state.kernel_mat[idx_i, idx_i]
            - state.y_bin[idx_j] * (state.alpha[idx_j] - aj_old) * state.kernel_mat[idx_i, idx_j]
        )
        b2 = (
            state.bias - err_j
            - state.y_bin[idx_i] * (state.alpha[idx_i] - ai_old) * state.kernel_mat[idx_i, idx_j]
            - state.y_bin[idx_j] * (state.alpha[idx_j] - aj_old) * state.kernel_mat[idx_j, idx_j]
        )

        # Prefer whichever candidate corresponds to a genuinely non-bound
        # (0 < alpha < C) sample — that sample's own KKT condition is the
        # one that actually constrains the bias; if both samples ended up
        # on the boundary, split the difference.
        if 0 < state.alpha[idx_i] < self._cfg.reg_param:
            new_bias = b1
        elif 0 < state.alpha[idx_j] < self._cfg.reg_param:
            new_bias = b2
        else:
            new_bias = 0.5 * (b1 + b2)

        # Update every sample's cached error incrementally, rather than
        # recomputing the whole error vector from scratch — the change in
        # error caused by this pair update is exactly the change in
        # alpha_i's contribution plus the change in alpha_j's contribution
        # plus the change in bias, applied to every sample via its own
        # kernel value against i and j. `state.buf` is reused as scratch
        # space (via the `out=` argument) to avoid allocating a new array
        # on every single pair update — this function runs extremely often.
        db = new_bias - state.bias
        np.multiply(
            state.kernel_mat[idx_i],
            float(state.y_bin[idx_i]) * (state.alpha[idx_i] - ai_old),
            out=state.buf,
        )
        state.errors += state.buf
        np.multiply(
            state.kernel_mat[idx_j],
            float(state.y_bin[idx_j]) * (state.alpha[idx_j] - aj_old),
            out=state.buf,
        )
        state.errors += state.buf
        state.errors += db
        state.bias = new_bias
        return True

    # Receives: idx_i — the "first" sample of the pair we're choosing a
    #   partner for, and state — the current SMO state.
    # Calculates: for every OTHER sample j, whether pairing it with idx_i
    #   would even have a non-empty feasible range for the analytic update
    #   (see the docstring's derivation) — samples where both alphas are
    #   already pinned to opposite constraint boundaries can never be
    #   usefully paired.
    # Returns: a boolean mask, one entry per sample, True wherever pairing
    #   with idx_i is at least mathematically possible (idx_i's own entry
    #   is always forced to False, since a sample can't pair with itself).
    def _valid_j_mask(self, idx_i: int, state: _SMOState) -> np.ndarray:
        """Boolean mask of j indices with non-degenerate box constraint (lower < upper).

        After all alphas saturate to 0 or C, the standard error-diff heuristic
        often selects pairs with lower == upper (e.g. alpha_i=0, alpha_j=C,
        different class).  This mask restricts selection to pairs where the
        SMO analytic step has a non-empty feasible interval.

        Derivation:
        - same class   valid iff 0 < alpha_i+alpha_j < 2C (not both-zero or both-C)
        - diff class   valid iff |alpha_i-alpha_j| < C  (not on opposite boundaries)
        """
        ai = state.alpha[idx_i]
        cap = self._cfg.reg_param
        same = state.y_bin == state.y_bin[idx_i]
        ssum = state.alpha + ai
        valid = np.where(same, (ssum > 0.0) & (ssum < 2.0 * cap),
                         np.abs(state.alpha - ai) < cap)
        valid[idx_i] = False
        return valid

    # Receives: idx_i — the sample SMO is currently trying to fix a KKT
    #   violation for, and state — the current SMO state.
    # Calculates: picks the best partner j to pair idx_i with, preferring
    #   (in order) a non-bound sample with the largest |error difference|
    #   from idx_i, then falling back to ANY sample with a mathematically
    #   valid (non-degenerate) pairing if no non-bound choice works — and
    #   attempts the actual pair update via `_try_pair` for whichever j was
    #   chosen.
    # Returns: True if some pair update was successfully applied; False if
    #   no usable partner could be found at all.
    def _try_best_j(self, idx_i: int, state: _SMOState) -> bool:
        """Select the best j and attempt a pair update.  Returns True on success.

        Heuristic priority:
        1. Non-bound (0 < alpha < C) j by max |error_diff|.
        2. Any j with a valid (non-degenerate) box constraint, by max |error_diff|.

        Priority 2 is the critical fallback: once all alphas saturate to the
        boundary (alpha=0 or alpha=C), the primary heuristic always picks a j
        whose box constraint is degenerate.  ``_valid_j_mask`` identifies the
        feasible candidates so the loop keeps making progress.
        """
        # A bigger gap between two samples' current errors tends to
        # produce a bigger, more useful update step, so candidates are
        # ranked by |error_i - error_j|.
        err_diff = np.abs(state.errors - state.errors[idx_i])
        err_diff[idx_i] = -1.0
        # Temporarily exclude idx_i from its own non-bound mask (a sample
        # can't be its own pairing partner) while checking the primary
        # heuristic.
        nb_i = bool(state.nb[idx_i])
        state.nb[idx_i] = False
        if state.nb.any():
            nb_idx = np.where(state.nb)[0]
            if self._try_pair(idx_i, int(nb_idx[np.argmax(err_diff[nb_idx])]), state):
                # idx_i's own non-bound status may have changed as a result
                # of the update — refresh it directly rather than waiting
                # for the next full `refresh_nb` call.
                state.nb[idx_i] = 0.0 < state.alpha[idx_i] < self._cfg.reg_param
                return True
        # Restore idx_i's original non-bound flag before trying the
        # fallback heuristic (it was only masked out above to keep it from
        # being selected as its own partner).
        state.nb[idx_i] = nb_i
        valid = self._valid_j_mask(idx_i, state)
        if valid.any():
            v_idx = np.where(valid)[0]
            return self._try_pair(idx_i, int(v_idx[np.argmax(err_diff[v_idx])]), state)
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Receives: x_train — the training feature matrix, and y_bin — the
    #   labels already encoded as {-1, +1}.
    # Calculates: runs the full SMO outer loop: repeatedly finds every
    #   sample that currently violates the KKT optimality conditions and
    #   tries to fix each one via a pair update, until either no violations
    #   remain, no update could be made in a full pass, or `max_iter`
    #   passes have run. Every 50 steps, the error vector is recomputed
    #   from scratch to guard against floating-point drift accumulated by
    #   the incremental updates.
    # Returns: self — stores the final support vectors (samples with
    #   alpha > 0, i.e. those that actually matter for prediction), their
    #   labels, feature rows, original indices, the fitted bias, and the
    #   precomputed alpha*y dual coefficients in `self._sv`.
    def fit(self, x_train: np.ndarray, y_bin: np.ndarray) -> "_BinarySVC":
        """Run SMO on x_train, y_bin in {-1, +1}.  Returns self for chaining."""
        kernel_mat = self._compute_kernel(x_train, x_train)
        # Diagonal jitter for numerical stability (precision handling).
        # Adding a tiny constant to the diagonal keeps the kernel matrix
        # well-conditioned (helps avoid `eta` landing exactly at 0 due to
        # floating-point rounding when two samples are near-identical).
        np.fill_diagonal(kernel_mat, kernel_mat.diagonal() + 1e-8)
        state = _SMOState(y_bin, kernel_mat)
        for step in range(self._cfg.max_iter):
            kkt_idx = state.kkt_violations(self._cfg.tol, self._cfg.reg_param)
            if len(kkt_idx) == 0:
                # No sample violates the optimality conditions anymore —
                # the solution has converged.
                break
            state.refresh_nb(self._cfg.reg_param)
            n_changed = 0
            for idx_arr in kkt_idx:
                if self._try_best_j(int(idx_arr), state):
                    n_changed += 1
            if n_changed == 0:
                # A full pass over every violator produced zero successful
                # updates — stuck, so stop rather than looping forever.
                break
            if (step + 1) % 50 == 0:
                state.recompute_errors()
        # Clip alphas to [0, C] (numerical stability).
        # The incremental updates should already respect the box
        # constraint, but a final clip guards against tiny floating-point
        # overshoots accumulating over many iterations.
        state.alpha = np.clip(state.alpha, 0.0, self._cfg.reg_param)
        # Any sample with an essentially-zero alpha contributes nothing to
        # predictions — only samples with alpha meaningfully above 0 are
        # kept as support vectors.
        sv_mask = state.alpha > 1e-5
        self._sv = _SVFit(
            alpha=state.alpha[sv_mask],
            sv_y=y_bin[sv_mask],
            sv_x=x_train[sv_mask],
            sv_idx=np.where(sv_mask)[0],
            bias=state.bias,
            dual_coef=(state.alpha[sv_mask] * y_bin[sv_mask]).reshape(1, -1),
        )
        return self

    # Receives: x_data — feature data to score (already scaled the same
    #   way training data was).
    # Calculates: the kernel value between every sample in x_data and every
    #   support vector, then combines those with the dual coefficients and
    #   bias to get the raw (unthresholded) SVM decision score.
    # Returns: an array of shape (n_samples,) — the signed distance from
    #   the decision boundary for every sample (positive means "positive
    #   class side", negative means "negative class side").
    def decision_function(self, x_data: np.ndarray) -> np.ndarray:
        """Raw decision scores for samples in x_data."""
        kernel_mat = self._compute_kernel(x_data, self.sv_x_)
        return kernel_mat @ (self.alpha_ * self.sv_y_) + self.b_

    # Receives: x_data — feature data to classify.
    # Calculates: the raw decision score for every sample (via
    #   `decision_function`), then just its sign.
    # Returns: an array of {-1, +1} predicted labels.
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """Predict binary labels in {-1, +1}."""
        return np.sign(self.decision_function(x_data))


class SVC:
    """Kernel Support Vector Classifier via SMO (dual formulation).

    Supports binary and multi-class classification (one-vs-one strategy).
    Kernels: ``'linear'``, ``'rbf'``, ``'poly'``.

    Parameters
    ----------
    kernel : {'linear', 'rbf', 'poly'}, default='rbf'
        Specifies the kernel function.
    reg_param : float, default=1.0
        Regularisation parameter (also accepted as keyword ``C=``).
        Larger value ↔ harder margin, less regularisation.
    gamma : float or {'scale', 'auto'}, default='scale'
        Kernel coefficient for ``'rbf'`` and ``'poly'``.
        ``'scale'`` → ``1 / (n_features * X.var())``;
        ``'auto'``  → ``1 / n_features``.
    **kwargs
        Additional keyword arguments: ``degree`` (int, default 3),
        ``coef0`` (float, default 1.0), ``tol`` (float, default 1e-3),
        ``max_iter`` (int, default 1000),
        ``random_state`` (int or None, default None).
        ``C`` is accepted as an alias for ``reg_param``.

    Attributes
    ----------
    support_vectors_ : np.ndarray of shape (n_sv, n_features)
        All support vectors across all binary classifiers.  Set after :meth:`fit`.
    support_ : np.ndarray of int, shape (n_sv,)
        Training indices of support vectors.  Set after :meth:`fit`.
    dual_coef_ : np.ndarray of shape (1, n_total_sv)
        Flattened dual coefficients from all binary classifiers.
        Set after :meth:`fit`.
    intercept_ : np.ndarray of shape (n_classifiers,)
        Bias terms from each binary classifier.  Set after :meth:`fit`.
    n_support_ : np.ndarray of int, shape (n_classes,)
        Approximate number of support vectors per class.  Set after :meth:`fit`.
    classes_ : np.ndarray of shape (n_classes,)
        Unique class labels encountered during :meth:`fit`.
    """

    # Receives: kernel, reg_param, gamma (given their own named
    #   parameters), plus **kwargs for degree, coef0, tol, max_iter,
    #   random_state, and the `C` alias for reg_param.
    # Calculates: resolves the `C` alias, validates every hyperparameter,
    #   and bundles kernel/gamma/degree/coef0/reg_param/tol/max_iter into
    #   one immutable `_BinaryConfig` (reused as-is for a binary problem,
    #   or as a template for every pairwise classifier in multiclass mode).
    # Returns: nothing — __init__ only prepares the object; `self._fit_state`
    #   starts as None.
    def __init__(
        self,
        kernel: str = "rbf",
        reg_param: float = 1.0,
        gamma: Union[str, float] = "scale",
        **kwargs,
    ) -> None:
        # Accept sklearn's C= kwarg as an alias for reg_param (error handling).
        reg_param = float(kwargs.pop("C", reg_param))
        degree: int = int(kwargs.pop("degree", 3))
        coef0: float = float(kwargs.pop("coef0", 1.0))
        tol: float = float(kwargs.pop("tol", 1e-3))
        max_iter: int = int(kwargs.pop("max_iter", 1000))
        random_state: Optional[int] = kwargs.pop("random_state", None)

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {list(kwargs)}")
        if reg_param <= 0:
            raise ValueError(f"reg_param (C) must be positive, got {reg_param}")
        if kernel not in ("linear", "rbf", "poly"):
            raise ValueError(
                f"Unknown kernel '{kernel}'. Supported: 'linear', 'rbf', 'poly'."
            )
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")

        self._cfg: _BinaryConfig = _BinaryConfig(
            kernel=kernel,
            gamma=gamma,  # type: ignore[arg-type]
            degree=degree,
            coef0=coef0,
            reg_param=reg_param,
            tol=tol,
            max_iter=max_iter,
        )
        self._random_state: Optional[int] = random_state
        self._fit_state: Optional[_SVCFitState] = None

    # ------------------------------------------------------------------
    # Backward-compatible attribute access
    # ------------------------------------------------------------------

    # Receives: name — the attribute name someone tried to access (Python
    #   only calls this once ordinary attribute lookup has already failed).
    # Calculates: checks whether `name` matches one of the config fields
    #   directly, or one of the sklearn-style aliases (`C`, `gamma`).
    # Returns: the resolved value from `self._cfg`; raises AttributeError
    #   if `name` isn't recognised.
    def __getattr__(self, name: str):
        """Provide sklearn-compatible attribute aliases (e.g. C, kernel, gamma)."""
        cfg_fields = ("kernel", "degree", "coef0", "tol", "max_iter")
        if name in cfg_fields and self._cfg is not None:
            return getattr(self._cfg, name)
        if name == "C" and self._cfg is not None:
            return self._cfg.reg_param
        if name == "gamma" and self._cfg is not None:
            return self._cfg.gamma
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    # ------------------------------------------------------------------
    # Public fitted-attribute properties
    # ------------------------------------------------------------------

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored support vectors.
    # Returns: every support vector across every binary classifier this
    #   SVC trained, or an empty (0, 0) array if not yet fitted.
    @property
    def support_vectors_(self) -> np.ndarray:
        """Support vectors from all binary classifiers."""
        fs = self._fit_state
        return fs.support_vectors if fs is not None else np.empty((0, 0))

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored support-vector indices.
    # Returns: the training-set row index of every support vector, or
    #   empty if not yet fitted.
    @property
    def support_(self) -> np.ndarray:
        """Training indices of support vectors."""
        fs = self._fit_state
        return fs.support if fs is not None else np.empty(0, dtype=int)

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored, flattened dual coefficients.
    # Returns: a (1, n_total_sv) row combining every binary classifier's
    #   own alpha*y values, or empty if not yet fitted.
    @property
    def dual_coef_(self) -> np.ndarray:
        """Dual coefficients flattened from all binary classifiers."""
        fs = self._fit_state
        return fs.dual_coef if fs is not None else np.empty((1, 0))

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored per-classifier bias terms.
    # Returns: one bias value per binary classifier trained, or [0.0] if
    #   not yet fitted.
    @property
    def intercept_(self) -> np.ndarray:
        """Bias terms from each binary classifier."""
        fs = self._fit_state
        return fs.intercept if fs is not None else np.array([0.0])

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored per-class support-vector
    #   counts.
    # Returns: an approximate support-vector count for each class, or
    #   empty if not yet fitted.
    @property
    def n_support_(self) -> np.ndarray:
        """Approximate support-vector count per class."""
        fs = self._fit_state
        return fs.n_support if fs is not None else np.empty(0, dtype=int)

    # Receives: nothing besides self.
    # Calculates: nothing — reads the stored class labels.
    # Returns: every distinct class label seen during fit(), or empty if
    #   not yet fitted.
    @property
    def classes_(self) -> np.ndarray:
        """Unique class labels seen at fit time."""
        return self._fit_state.classes if self._fit_state else np.empty(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Receives: x_data — the training feature matrix, and y — the class
    #   labels (any number of distinct values).
    # Calculates: standardises the features (after imputing any missing
    #   values), resolves 'scale'/'auto' gamma strings into a concrete
    #   number, and then — depending on how many distinct classes are
    #   present — either does nothing further (a single class has nothing
    #   to learn to separate), trains ONE binary SMO classifier (exactly 2
    #   classes), or trains one binary classifier per PAIR of classes
    #   (One-vs-One, for 3+ classes).
    # Returns: self, storing the fitted classifiers, their class pairings,
    #   the resolved gamma, the combined support vectors, and the scaler
    #   statistics in `self._fit_state`.
    def fit(self, x_data: np.ndarray, y: np.ndarray) -> "SVC":
        """Train the SVC.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        x_data, y, classes = prepare_fit_inputs(x_data, y)

        if len(x_data) == 0:
            self._fit_state = self._empty_fit_state(x_data, classes)
            return self

        x_data, _ = impute_column_medians(x_data)

        # Standardise every feature to zero mean / unit variance — kernel
        # methods (especially RBF) are sensitive to the raw scale of the
        # input features.
        scaler_mean = x_data.mean(axis=0)
        scaler_scale = np.maximum(x_data.std(axis=0), 1e-8)
        scaled = _ScaledData(
            x_scaled=(x_data - scaler_mean) / scaler_scale,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
        )

        # Resolve gamma='scale'/'auto' into an actual number now, using the
        # SCALED data (matching sklearn's own convention).
        gamma_val = resolve_gamma(self._cfg.gamma, scaled.x_scaled)

        if len(classes) == 1:
            # Only one class ever appeared — there's no separating
            # boundary to learn; predict() will just always return that
            # one class.
            self._fit_state = self._single_class_fit_state(x_data, classes)
            return self

        cfg = _BinaryConfig(
            kernel=self._cfg.kernel,
            gamma=gamma_val,
            degree=self._cfg.degree,
            coef0=self._cfg.coef0,
            reg_param=self._cfg.reg_param,
            tol=self._cfg.tol,
            max_iter=self._cfg.max_iter,
        )

        try:
            if len(classes) == 2:
                batch = self._fit_binary_pair(scaled.x_scaled, y, classes, cfg)
            else:
                batch = self._fit_ovo(scaled.x_scaled, y, classes, cfg)
        except (ValueError, np.linalg.LinAlgError, MemoryError) as exc:
            raise RuntimeError(f"SVC fitting failed: {exc}") from exc

        self._fit_state = self._collect_fit_state(scaled, batch, gamma_val, classes)
        return self

    # Receives: x_data and y — a dataset to split into pieces, and
    #   chunk_size — the target size of each piece.
    # Calculates: nothing — a generator that produces each chunk lazily,
    #   one at a time.
    # Returns (yields): successive (x_chunk, y_chunk) pairs covering the
    #   whole dataset, in order.
    @staticmethod
    def _batch_generator(x_data: np.ndarray, y: np.ndarray, chunk_size: int = 32):
        """Yield (x_chunk, y_chunk) mini-batches from a data stream.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        chunk_size : int, default=32

        Yields
        ------
        tuple
            ``(x_chunk, y_chunk)`` mini-batch pair for data streaming.
        """
        n_samples = len(x_data)
        for start in range(0, n_samples, chunk_size):
            yield x_data[start:start + chunk_size], y[start:start + chunk_size]

    # Receives: x_data and y — new data arriving incrementally, and
    #   chunk_size — how many samples to process per mini-batch.
    # Calculates: nothing new of its own — simply calls the ordinary
    #   `fit()` once per mini-batch (this SVC implementation doesn't have a
    #   true incremental SMO update, so each "partial" step is really a
    #   full refit on just that chunk).
    # Returns: self, updated after processing every chunk in turn.
    def partial_fit(
        self, x_data: np.ndarray, y: np.ndarray, chunk_size: int = 32
    ) -> "SVC":
        """Refit on each chunk of incoming data (data streaming).

        Splits the data into mini-batches using :meth:`_batch_generator` and
        refits on each chunk in turn.  Use this method to integrate SVC into
        data streaming pipelines where labelled data arrives incrementally.

        Parameters
        ----------
        x_data : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        chunk_size : int, default=32
            Number of samples per mini-batch.

        Returns
        -------
        self
        """
        x_data = np.asarray(x_data, dtype=np.float64)
        for x_b, y_b in self._batch_generator(x_data, y, chunk_size):
            self.fit(x_b, y_b)
        return self

    # Receives: x_data — new feature data to classify.
    # Calculates: imputes/scales x_data the same way training data was
    #   handled, then — depending on how many classes were seen at fit
    #   time — returns the single known class, thresholds the one binary
    #   classifier's decision score at 0, or runs the One-vs-One majority
    #   vote across every pairwise classifier.
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
        x_data = np.asarray(x_data, dtype=np.float64)
        if x_data.ndim == 1:
            x_data = x_data.reshape(1, -1)
        x_data = self._impute(x_data)

        if self._fit_state is None:
            raise ValueError("SVC is not fitted yet. Call fit() first.")

        if self._fit_state.scaler_mean is not None:
            x_data = (x_data - self._fit_state.scaler_mean) / self._fit_state.scaler_scale

        if self._fit_state.single_class is not None:
            # Only one class was ever seen at fit time — every prediction
            # is trivially that same class.
            return np.full(len(x_data), self._fit_state.single_class)

        if len(self._fit_state.classes) == 2:
            raw = self._fit_state.classifiers[0].decision_function(x_data)
            return np.where(raw >= 0, self._fit_state.classes[1], self._fit_state.classes[0])

        return self._ovo_predict(x_data)

    # Receives: x_data — new feature data to score.
    # Calculates: the same imputation/scaling as `predict`, then either the
    #   single binary classifier's raw decision score, or every pairwise
    #   classifier's own raw score stacked into one table.
    # Returns: for binary classification, an array of shape (n_samples,);
    #   for multiclass (OvO), an array of shape (n_samples, n_classifiers).
    def decision_function(self, x_data: np.ndarray) -> np.ndarray:
        """Compute decision scores.

        For binary classification returns shape ``(n_samples,)``.
        For multi-class (OvO) returns shape ``(n_samples, n_classifiers)``.

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
        x_data = self._impute(x_data)

        if self._fit_state is None:
            raise ValueError("SVC is not fitted yet. Call fit() first.")

        if self._fit_state.scaler_mean is not None:
            x_data = (x_data - self._fit_state.scaler_mean) / self._fit_state.scaler_scale

        clfs = self._fit_state.classifiers
        if len(self._fit_state.classes) == 2:
            return clfs[0].decision_function(x_data)

        # One column per pairwise classifier, side by side.
        return np.column_stack([clf.decision_function(x_data) for clf in clfs])

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Receives: cfg — the bundled hyperparameters for one binary
    #   classifier.
    # Calculates: nothing — just constructs a fresh `_BinarySVC` with this
    #   SVC's own random_state.
    # Returns: a new, not-yet-fitted `_BinarySVC` instance.
    def _make_binary_clf(self, cfg: _BinaryConfig) -> _BinarySVC:
        return _BinarySVC(cfg, random_state=self._random_state)

    # Receives: x_train — the (scaled) training data, y_labels — the
    #   original class labels, classes — the (exactly 2) distinct class
    #   values, and cfg — the binary classifier's hyperparameters.
    # Calculates: encodes the two classes as {-1, +1} (classes[0] -> -1,
    #   classes[1] -> +1) and trains one `_BinarySVC` on the whole dataset.
    # Returns: a `_ClassifierBatch` containing that single fitted
    #   classifier and its one class pair.
    def _fit_binary_pair(self, x_train, y_labels, classes, cfg) -> _ClassifierBatch:
        """Train one binary classifier; return a _ClassifierBatch."""
        mask = (y_labels == classes[0]) | (y_labels == classes[1])
        x_sub = x_train[mask]
        y_sub = np.where(y_labels[mask] == classes[0], -1.0, 1.0)
        clf = self._make_binary_clf(cfg)
        clf.fit(x_sub, y_sub)
        return _ClassifierBatch([clf], [(classes[0], classes[1])])

    # Receives: x_train — the (scaled) training data, y_labels — the
    #   original class labels, classes — every distinct class (3 or more),
    #   and cfg — the binary classifier hyperparameters shared by every
    #   pairwise classifier.
    # Calculates: for every possible PAIR of classes, selects just the
    #   training samples belonging to that pair, encodes them as {-1, +1},
    #   and trains one independent `_BinarySVC` — building up
    #   C(n_classes, 2) classifiers total (e.g. 3 classes -> 3 classifiers,
    #   4 classes -> 6 classifiers).
    # Returns: a `_ClassifierBatch` containing every one of those pairwise
    #   classifiers alongside the (negative_class, positive_class) pair
    #   each one was trained on.
    def _fit_ovo(self, x_train, y_labels, classes, cfg) -> _ClassifierBatch:
        """One-vs-One: train C(k,2) classifiers (independent, supports parallel processing)."""
        classifiers = []
        pairs = []
        for neg_cls, pos_cls in combinations(classes, 2):
            mask = (y_labels == neg_cls) | (y_labels == pos_cls)
            x_sub = x_train[mask]
            y_sub = np.where(y_labels[mask] == neg_cls, -1.0, 1.0)
            clf = self._make_binary_clf(cfg)
            clf.fit(x_sub, y_sub)
            classifiers.append(clf)
            pairs.append((neg_cls, pos_cls))
        return _ClassifierBatch(classifiers, pairs)

    # Receives: x_data — feature data to classify (already scaled).
    # Calculates: for every one of the pairwise classifiers, whether it
    #   voted for its own "positive" or "negative" class on each sample,
    #   and tallies up votes per (sample, class) — all at once via array
    #   operations rather than looping over samples.
    # Returns: an array of shape (n_samples,) — for each sample, whichever
    #   class received the most votes across all pairwise classifiers.
    def _ovo_predict(self, x_data: np.ndarray) -> np.ndarray:
        """Majority vote across all OvO classifiers.

        Vote accumulation is fully vectorised — no Python loop over samples.
        """
        if self._fit_state is None:
            raise ValueError("SVC is not fitted yet. Call fit() first.")
        num_samples = len(x_data)
        classes = self._fit_state.classes
        num_classes = len(classes)
        # A dict mapping each class LABEL to its column position in the
        # `votes` table below.
        class_idx = {c: i for i, c in enumerate(classes)}
        votes: np.ndarray = np.zeros((num_samples, num_classes), dtype=np.intp)

        for clf, (neg_cls, pos_cls) in zip(
            self._fit_state.classifiers, self._fit_state.pairs
        ):
            raw = clf.decision_function(x_data)
            # For every sample, did this pairwise classifier side with its
            # "positive" class or its "negative" class?
            pos_wins = raw >= 0
            votes[:, class_idx[pos_cls]] += pos_wins.astype(np.intp)
            votes[:, class_idx[neg_cls]] += (~pos_wins).astype(np.intp)

        # Whichever class column has the most votes, per sample, is the
        # predicted class.
        return classes[np.argmax(votes, axis=1)]

    # Receives: scaled — the scaled training data plus its scaler
    #   statistics, batch — every fitted binary classifier and its class
    #   pairing, gamma_val — the resolved gamma value used, and classes —
    #   every distinct class label.
    # Calculates: combines the individual support-vector sets from every
    #   binary classifier into one overall (deduplicated) support-vector
    #   set, and flattens every classifier's own dual coefficients and
    #   bias into single combined arrays.
    # Returns: a fully populated `_SVCFitState` bundling everything
    #   `predict`/`decision_function`/the various properties need.
    def _collect_fit_state(
        self,
        scaled: _ScaledData,
        batch: _ClassifierBatch,
        gamma_val: float,
        classes: np.ndarray,
    ) -> _SVCFitState:
        """Build the immutable fit-state from trained binary classifiers."""
        x_train = scaled.x_scaled
        classifiers, pairs = batch.classifiers, batch.pairs
        # A sample might be a support vector for MORE THAN ONE pairwise
        # classifier (in OvO mode); np.unique deduplicates the combined
        # list of support-vector indices from every classifier down to one
        # entry per distinct sample.
        all_sv_idx = np.unique(
            np.concatenate([clf.sv_idx_ for clf in classifiers])
            if classifiers else np.array([], dtype=int)
        )
        support_vectors = x_train[all_sv_idx] if len(all_sv_idx) > 0 else x_train[:0]

        n_sv_per_class: np.ndarray = np.zeros(len(classes), dtype=int)
        if len(all_sv_idx) > 0:
            n_sv_per_class[:] = max(1, len(all_sv_idx) // len(classes))

        intercept = np.array([clf.b_ for clf in classifiers])

        if classifiers:
            # Every classifier's own (1, n_sv_for_that_pair) dual
            # coefficient row, flattened and concatenated end to end into
            # one long combined row.
            flat = np.concatenate([clf.dual_coef_.ravel() for clf in classifiers])
            dual_coef = flat.reshape(1, -1)
        else:
            dual_coef = np.empty((1, 0))

        return _SVCFitState(
            classifiers=classifiers,
            pairs=pairs,
            gamma_val=gamma_val,
            single_class=None,
            support_vectors=support_vectors,
            support=all_sv_idx,
            dual_coef=dual_coef,
            intercept=intercept,
            n_support=n_sv_per_class,
            classes=classes,
            scaler_mean=scaled.scaler_mean,
            scaler_scale=scaled.scaler_scale,
        )

    # Receives: x_data — the (empty) training data, and classes — whatever
    #   (possibly empty) class labels were found.
    # Calculates: nothing — just builds a placeholder "nothing was trained"
    #   fit-state for the degenerate empty-dataset case.
    # Returns: a `_SVCFitState` with every field set to an empty/trivial
    #   default.
    def _empty_fit_state(
        self, x_data: np.ndarray, classes: np.ndarray
    ) -> _SVCFitState:
        return _SVCFitState(
            classifiers=[],
            pairs=[],
            gamma_val=1.0,
            single_class=None,
            support_vectors=x_data[:0],
            support=np.array([], dtype=int),
            dual_coef=np.empty((1, 0)),
            intercept=np.array([0.0]),
            n_support=np.zeros(len(classes), dtype=int),
            classes=classes,
            scaler_mean=None,
            scaler_scale=None,
        )

    # Receives: x_data — the training data, and classes — the single
    #   distinct class label that was found.
    # Calculates: nothing — builds a placeholder fit-state marking that
    #   only one class exists, so predict() can shortcut straight to that
    #   answer.
    # Returns: a `_SVCFitState` with `single_class` set to that one label
    #   and every other field trivial/empty.
    def _single_class_fit_state(
        self, x_data: np.ndarray, classes: np.ndarray
    ) -> _SVCFitState:
        return _SVCFitState(
            classifiers=[],
            pairs=[],
            gamma_val=1.0,
            single_class=classes[0],
            support_vectors=x_data[:0],
            support=np.array([], dtype=int),
            dual_coef=np.empty((1, 0)),
            intercept=np.array([0.0]),
            n_support=np.zeros(len(classes), dtype=int),
            classes=classes,
            scaler_mean=None,
            scaler_scale=None,
        )

    # Receives: x_data — feature data that may contain NaN values.
    # Calculates: if there are no NaNs, nothing; otherwise replaces every
    #   NaN with a flat 0.0.
    # Returns: the NaN-free array (a fresh copy if anything was imputed, or
    #   the original array unchanged if there was nothing to fix).
    @staticmethod
    def _impute(x_data: np.ndarray) -> np.ndarray:
        """Replace NaN with 0 at prediction time."""
        if not np.isnan(x_data).any():
            return x_data
        x_clean = x_data.copy()
        x_clean[np.isnan(x_clean)] = 0.0
        return x_clean
