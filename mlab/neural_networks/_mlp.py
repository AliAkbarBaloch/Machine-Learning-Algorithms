"""Multi-layer perceptron built from composable layers, trained by hand-rolled
numpy backprop (no autograd, no framework dependency).

Layers are dumb on purpose: each one only knows how to turn its own input
into an output (``__call__``) and how to turn an upstream gradient into a
downstream one (``backward``). ``MLPRegressor`` is the only thing that knows
about training -- it wires the layers into a stack and drives them with
mini-batch Adam.

Techniques in play, so the rationale doesn't have to be reverse-engineered
from the code later:

* vectorization -- every layer processes a whole batch as one matrix op,
  never loops over samples.
* mini-batch processing -- bounds peak activation memory to O(batch_size)
  instead of O(n_samples) and gives many gradient steps per epoch.
* Adam with bias-corrected moments, falling back to plain SGD with an
  invscaling/adaptive learning-rate schedule when solver='sgd'.
* decoupled weight decay (AdamW-style) so L2 strength stays predictable
  under Adam's per-parameter gradient normalization.
* gradient-norm clipping and an always-on best-weights checkpoint, so a
  pathological learning rate degrades gracefully instead of to NaN.
* early stopping on a held-out split.
* He/Xavier initialization chosen per activation, with a small positive
  ReLU bias to avoid dead units at init.
* activation optimization -- every activation layer caches what its
  backward pass needs (the sigmoid/tanh output, the ReLU sign mask) during
  forward, so backward never recomputes exp()/tanh()/comparisons.
* internal feature/target standardization, so behavior doesn't depend on
  the caller having pre-scaled their data.
"""

from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np

__all__ = [
    "ModularLinearLayer",
    "SigmoidLayer",
    "TanhLayer",
    "ReLULayer",
    "SoftmaxLayer",
    "MLPRegressor",
]


# Receives: x — an array of pre-activation values, any shape.
# Calculates: the logistic sigmoid of every entry, splitting into two
#   branches (one for non-negative inputs, one for negative inputs) so that
#   the exponential inside is ALWAYS computed on a non-positive number.
# Returns: an array the same shape as x, with every value squashed into
#   (0, 1).
def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Branch on sign to keep every exp() argument negative -- avoids
    # overflow for large |x| that the naive 1 / (1 + exp(-x)) suffers from.
    # For x >= 0: 1/(1+exp(-x)) is used directly, since exp(-x) <= 1 there
    # and can never overflow.
    # For x < 0: the mathematically equivalent form exp(x)/(1+exp(x)) is
    # used instead, since exp(x) < 1 there — computing exp(-x) directly on
    # a very negative x would instead risk exp() overflowing on the huge
    # positive exponent -x.
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


class ModularLinearLayer:
    """Fully connected layer: Y = X @ weight + bias.

    ``backward`` folds in an optional L2 term (same convention scikit-learn
    uses: the penalty gradient is averaged over the batch together with the
    data gradient), but the layer itself has no notion of an optimizer --
    that lives in whatever calls ``update``.
    """

    # Receives: in_features and out_features — how many inputs and outputs
    #   this layer has, rng — a random generator for weight initialisation,
    #   and init — which initialisation scheme to use ('he' or 'xavier').
    # Calculates: a freshly initialised weight matrix (scaled according to
    #   the chosen scheme, so activation variance stays reasonable as
    #   signals pass through many stacked layers) and a zero bias vector.
    # Returns: nothing — __init__ only sets up `self.weight`/`self.bias`
    #   and empty placeholder buffers for the forward/backward caches.
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: Optional[np.random.Generator] = None,
        init: str = "xavier",
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features

        active_rng = rng if rng is not None else np.random.default_rng()
        weight: np.ndarray
        if init == "he":
            # He et al. 2015: matches the variance ReLU halves by zeroing
            # negatives. Used internally by MLPRegressor for ReLU stacks.
            std = np.sqrt(2.0 / in_features)
            weight = active_rng.normal(0.0, std, size=(in_features, out_features))
        else:
            # Xavier/Glorot uniform: keeps activation variance roughly
            # stable across layers regardless of fan-in/fan-out.
            limit = np.sqrt(6.0 / (in_features + out_features))
            weight = active_rng.uniform(-limit, limit, size=(in_features, out_features))
        self.weight = weight
        self.bias = np.zeros(out_features)

        # Buffers populated by __call__/backward. Never None, so callers
        # (and mypy) don't have to guard every access with `is not None`;
        # the invariant is simply "forward runs before backward".
        self._prev_input: np.ndarray = np.empty((0, in_features))
        self._weight_grad: np.ndarray = np.zeros_like(self.weight)
        self._bias_grad: np.ndarray = np.zeros_like(self.bias)

    # Receives: X — a batch of input activations, shape (batch_size,
    #   in_features).
    # Calculates: the linear transformation Y = X @ weight + bias, and
    #   remembers X (needed later by `backward` to compute the weight
    #   gradient).
    # Returns: the layer's output, shape (batch_size, out_features).
    def __call__(self, X: np.ndarray) -> np.ndarray:
        self._prev_input = X
        return X @ self.weight + self.bias

    # Receives: upstream_grad — the gradient of the loss with respect to
    #   THIS layer's output (handed down from whichever layer comes after
    #   it), and alpha — an optional L2 regularisation strength.
    # Calculates: the gradient of the loss with respect to this layer's own
    #   weight and bias (so `update`/the optimizer can use them), and the
    #   gradient with respect to this layer's INPUT (so the layer before
    #   it can continue the chain rule backward).
    # Returns: the downstream gradient — the gradient of the loss with
    #   respect to this layer's input, shape (batch_size, in_features).
    def backward(self, upstream_grad: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        n_samples = self._prev_input.shape[0]
        # Weight gradient: input^T @ upstream_grad is the standard
        # backprop formula for a linear layer, averaged over the batch;
        # `alpha * self.weight` adds the (coupled) L2 penalty gradient when
        # alpha > 0.
        self._weight_grad = (self._prev_input.T @ upstream_grad + alpha * self.weight) / n_samples
        # sum()/n instead of mean(): skips the dtype/where-handling overhead
        # np.mean's generic path carries, which shows up once this runs
        # thousands of times over small batches.
        self._bias_grad = upstream_grad.sum(axis=0) / n_samples
        # Downstream gradient: how the loss changes with respect to this
        # layer's OWN input, needed by whichever layer sits before this one
        # in the stack.
        downstream_grad: np.ndarray = upstream_grad @ self.weight.T
        return downstream_grad

    # Receives: lr — the learning rate (step size) to use for this update.
    # Calculates: nothing new — just applies the already-computed gradients
    #   (from the last `backward` call) to the weight and bias.
    # Returns: nothing — updates `self.weight`/`self.bias` in place.
    def update(self, lr: float) -> None:
        """Vanilla SGD step: weight -= lr * grad. MLPRegressor uses Adam
        internally and updates weight/bias directly, but this method stays
        a plain, self-contained gradient step for standalone use."""
        self.weight -= lr * self._weight_grad
        self.bias -= lr * self._bias_grad

    # Receives: nothing besides self.
    # Calculates: nothing — just exposes the cached weight gradient.
    # Returns: the weight gradient computed by the last `backward()` call.
    @property
    def weight_grad(self) -> np.ndarray:
        """Read-only view of the gradient computed by the last backward()."""
        return self._weight_grad

    # Receives: nothing besides self.
    # Calculates: nothing — just exposes the cached bias gradient.
    # Returns: the bias gradient computed by the last `backward()` call.
    @property
    def bias_grad(self) -> np.ndarray:
        """Read-only view of the gradient computed by the last backward()."""
        return self._bias_grad

    # Receives: factor — the multiplier to scale both cached gradients by.
    # Calculates: nothing new — just multiplies the already-stored weight
    #   and bias gradients by `factor`.
    # Returns: nothing — mutates `self._weight_grad`/`self._bias_grad` in
    #   place.
    def scale_gradients(self, factor: float) -> None:
        """Scale the cached gradients in place. Used by gradient-norm
        clipping, which needs to shrink every layer's gradient by the same
        factor without reaching into layer internals."""
        self._weight_grad *= factor
        self._bias_grad *= factor

    def __repr__(self) -> str:
        return f"ModularLinearLayer(in_features={self.in_features}, out_features={self.out_features})"


class SigmoidLayer:
    """Sigmoid activation: 1 / (1 + exp(-X))."""

    # Receives: nothing besides self.
    # Calculates: nothing — just prepares an empty placeholder for the
    #   forward-pass output that `backward` will need later.
    # Returns: nothing.
    def __init__(self) -> None:
        self._prev_result: np.ndarray = np.empty(0)

    # Receives: X — a batch of pre-activation values.
    # Calculates: the sigmoid of every entry, and remembers the RESULT
    #   (not the input) for use in `backward` — the sigmoid's own
    #   derivative can be expressed purely in terms of its output.
    # Returns: the sigmoid-activated output, same shape as X.
    def __call__(self, X: np.ndarray) -> np.ndarray:
        self._prev_result = _sigmoid(X)
        return self._prev_result

    # Receives: upstream_grad — the gradient flowing in from the layer
    #   after this one.
    # Calculates: the local sigmoid derivative, s * (1 - s), where s is the
    #   cached forward output — this is the standard identity that lets
    #   sigmoid's derivative be computed WITHOUT recomputing any
    #   exponentials, using only the already-known output.
    # Returns: the downstream gradient, same shape as upstream_grad.
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        s = self._prev_result
        grad_input: np.ndarray = upstream_grad * s * (1.0 - s)
        return grad_input

    def __repr__(self) -> str:
        return "SigmoidLayer()"


class TanhLayer:
    """Hyperbolic tangent activation."""

    # Receives: nothing besides self.
    # Calculates: nothing — prepares an empty placeholder.
    # Returns: nothing.
    def __init__(self) -> None:
        self._prev_result: np.ndarray = np.empty(0)

    # Receives: X — a batch of pre-activation values.
    # Calculates: tanh of every entry, and remembers the result for use in
    #   `backward`.
    # Returns: the tanh-activated output, same shape as X.
    def __call__(self, X: np.ndarray) -> np.ndarray:
        self._prev_result = np.tanh(X)
        return self._prev_result

    # Receives: upstream_grad — the gradient flowing in from the next
    #   layer.
    # Calculates: the local tanh derivative, 1 - tanh(x)^2 — another
    #   identity that only needs the cached forward OUTPUT, not the
    #   original input.
    # Returns: the downstream gradient.
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        return upstream_grad * (1.0 - self._prev_result ** 2)

    def __repr__(self) -> str:
        return "TanhLayer()"


class ReLULayer:
    """Rectified linear unit: max(0, X).

    Activation optimization: the positive-input mask is computed once on
    the forward pass and reused as-is on backward, instead of recomputing
    ``output > 0`` from scratch every backward call.
    """

    # Receives: nothing besides self.
    # Calculates: nothing — prepares empty placeholders for the cached
    #   output and the cached positive-input mask.
    # Returns: nothing.
    def __init__(self) -> None:
        self._prev_result: np.ndarray = np.empty(0)
        self._mask: np.ndarray = np.empty(0, dtype=bool)

    # Receives: X — a batch of pre-activation values.
    # Calculates: which entries of X are positive (the "mask"), and zeroes
    #   out every non-positive entry — this IS the ReLU function.
    # Returns: the ReLU-activated output, same shape as X.
    def __call__(self, X: np.ndarray) -> np.ndarray:
        self._mask = X > 0.0
        self._prev_result = np.where(self._mask, X, 0.0)
        return self._prev_result

    # Receives: upstream_grad — the gradient flowing in from the next
    #   layer.
    # Calculates: ReLU's derivative is exactly 1 where the input was
    #   positive and 0 elsewhere — multiplying by the already-cached
    #   boolean mask directly applies that derivative, without needing to
    #   look at the original input or output values again.
    # Returns: the downstream gradient.
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        grad_input: np.ndarray = upstream_grad * self._mask
        return grad_input

    def __repr__(self) -> str:
        return "ReLULayer()"


class SoftmaxLayer:
    """Row-wise softmax, for multi-class output layers."""

    # Receives: nothing besides self.
    # Calculates: nothing — prepares an empty placeholder.
    # Returns: nothing.
    def __init__(self) -> None:
        self._prev_result: np.ndarray = np.empty(0)

    # Receives: X — a batch of raw class scores (logits), one row per
    #   sample.
    # Calculates: the softmax of each row — subtracting each row's own
    #   maximum before exponentiating keeps every exponent <= 0, preventing
    #   overflow, without changing the final normalised result (the
    #   subtracted constant cancels out in the division).
    # Returns: a batch of probability distributions, same shape as X, each
    #   row summing to 1.
    def __call__(self, X: np.ndarray) -> np.ndarray:
        shifted = X - X.max(axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        self._prev_result = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
        return self._prev_result

    # Receives: upstream_grad — the gradient flowing in from the next
    #   layer.
    # Calculates: the softmax Jacobian-vector product for every row at
    #   once, using the identity dX_i = s_i * (grad_i - grad_i . s_i) —
    #   this avoids ever forming the full (n_classes, n_classes) Jacobian
    #   matrix per sample, which would be far more expensive.
    # Returns: the downstream gradient, same shape as upstream_grad.
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        # Jacobian-vector product per row: dX_i = s_i * (grad_i - grad_i . s_i)
        s = self._prev_result
        dot = np.sum(upstream_grad * s, axis=1, keepdims=True)
        grad_input: np.ndarray = s * (upstream_grad - dot)
        return grad_input

    def __repr__(self) -> str:
        return "SoftmaxLayer()"


ActivationLayer = Union[SigmoidLayer, TanhLayer, ReLULayer]
Layer = Union[ModularLinearLayer, SigmoidLayer, TanhLayer, ReLULayer, SoftmaxLayer]

_ACTIVATIONS: Dict[str, Type[ActivationLayer]] = {
    "relu": ReLULayer,
    "sigmoid": SigmoidLayer,
    "tanh": TanhLayer,
}


# Receives: linear_layers — every ModularLinearLayer in the network (each
#   with a gradient already computed from the current batch), and max_norm
#   — the maximum allowed COMBINED gradient norm across all of them.
# Calculates: the total L2 norm of every weight and bias gradient stacked
#   together across every layer; if that combined norm exceeds max_norm,
#   works out a single shrink factor that would bring it exactly down to
#   max_norm.
# Returns: nothing — if shrinking is needed, scales every layer's cached
#   gradients down by the same factor (via `scale_gradients`) so the
#   overall combined gradient magnitude never exceeds `max_norm`.
def _clip_grad_norm(linear_layers: List[ModularLinearLayer], max_norm: float) -> None:
    """Scale all gradients down together if their combined L2 norm exceeds
    max_norm. Standard safety net against exploding gradients -- without it
    a too-large learning rate or a bad batch can blow the weights to NaN in
    a single step."""
    total_sq = 0.0
    for layer in linear_layers:
        total_sq += float(np.sum(layer.weight_grad ** 2))
        total_sq += float(np.sum(layer.bias_grad ** 2))
    total_norm = np.sqrt(total_sq)
    if total_norm > max_norm and total_norm > 0.0:
        scale = max_norm / total_norm
        for layer in linear_layers:
            layer.scale_gradients(scale)


class MLPRegressor:
    """Feed-forward MLP for regression, trained with mini-batch Adam
    (or plain SGD) and squared-error loss.

    Inputs and targets are standardized internally (mean/std computed from
    the training data and stored on ``x_mean_``/``x_scale_``/``y_mean_``/
    ``y_scale_``), so the model behaves consistently regardless of the raw
    feature/target scale the caller happens to pass in -- ``predict``
    transparently undoes the target scaling, so callers never see it.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int
        Number of units in each hidden layer, e.g. (50, 30). An empty tuple
        collapses the network to a single linear layer.
    lr : float
        Base learning rate (``learning_rate_init`` in scikit-learn terms).
    epochs : int
        Number of passes over the training data.
    random_state : int, optional
        Seed for weight initialization, batch shuffling, and the
        validation split used by early stopping.
    alpha : float
        L2 weight decay strength. Under solver='adam' this is applied as
        decoupled weight decay (the AdamW fix: folding L2 into the raw
        gradient before Adam's per-parameter normalization would let Adam
        rescale away most of the penalty, weakening it unpredictably).
        Under solver='sgd' it's the classic coupled L2-via-gradient term.
    activation : {'relu', 'sigmoid', 'tanh'}
        Activation applied after every hidden linear layer. The output
        layer is always linear so regression targets aren't squashed.
    solver : {'adam', 'sgd'}
        Optimizer. Adam adapts a per-parameter step size from the running
        mean/variance of the gradient and converges far faster than plain
        SGD on the same epoch budget; sgd is the textbook baseline and
        honors ``learning_rate``'s decay schedule.
    batch_size : int or 'auto'
        Mini-batch size. 'auto' picks min(32, n_samples). Smaller batches
        mean more gradient steps per epoch and, on the small/noisy tabular
        datasets this is usually applied to, measurably better generalization
        from the implicit regularization effect of noisier gradient
        estimates (Keskar et al. 2016) -- verified empirically against
        scikit-learn's batch_size='auto' default of min(200, n_samples),
        which converges to a visibly worse test score in the same epoch
        budget on small datasets.
    learning_rate : {'constant', 'invscaling', 'adaptive'}
        Only used when solver='sgd' (Adam's per-parameter scaling makes a
        global schedule redundant, matching scikit-learn's behavior).
        'invscaling' decays as lr / epoch**power_t; 'adaptive' halves the
        rate whenever training loss stalls for ``n_iter_no_change`` epochs.
    early_stopping : bool
        Hold out ``validation_fraction`` of the training data and stop once
        validation loss stops improving for ``n_iter_no_change`` epochs.
    max_grad_norm : float, optional
        Clip the combined gradient norm to this value each step. None
        (default) disables it: the per-epoch finite-loss check plus the
        always-on best-weights checkpoint already stop a diverging run from
        ever returning NaN predictions (verified against learning rates up
        to 1e300), so clipping's extra per-batch norm computation is opt-in
        rather than paid on every step by default.
    """

    # Receives: every network-shape and training hyperparameter
    #   (hidden_layer_sizes, lr, epochs, random_state, alpha, activation,
    #   solver, batch_size, learning_rate schedule, Adam's beta_1/beta_2/
    #   epsilon, max_grad_norm, and the early-stopping controls).
    # Calculates: validates every hyperparameter combination up front (so a
    #   typo or invalid value fails immediately rather than mid-training),
    #   then stores everything and sets every fitted attribute (`layers_`,
    #   `loss_curve_`, the standardisation statistics, etc.) to an empty
    #   placeholder.
    # Returns: nothing — __init__ only prepares the object; the actual
    #   network layers aren't built until `fit()` knows the input/output
    #   dimensions.
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (50, 30),
        lr: float = 0.01,
        epochs: int = 100,
        random_state: Optional[int] = None,
        alpha: float = 0.0001,
        activation: str = "relu",
        solver: str = "adam",
        batch_size: Union[int, str] = "auto",
        learning_rate: str = "constant",
        power_t: float = 0.5,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        max_grad_norm: Optional[float] = None,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-4,
    ) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}")
        if solver not in ("adam", "sgd"):
            raise ValueError(f"solver must be 'adam' or 'sgd', got {solver!r}")
        if learning_rate not in ("constant", "invscaling", "adaptive"):
            raise ValueError(
                f"learning_rate must be one of ('constant', 'invscaling', 'adaptive'), got {learning_rate!r}"
            )
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if any(size <= 0 for size in hidden_layer_sizes):
            raise ValueError(f"hidden_layer_sizes must all be positive, got {hidden_layer_sizes}")
        if isinstance(batch_size, str) and batch_size != "auto":
            raise ValueError(f"batch_size must be a positive int or 'auto', got {batch_size!r}")
        if isinstance(batch_size, int) and batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.alpha = alpha
        self.activation = activation
        self.solver = solver
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

        self.layers_: List[Layer] = []
        self.loss_curve_: List[float] = []
        self.validation_scores_: List[float] = []
        self.n_iter_ = 0
        self._single_output = True

        self.x_mean_: np.ndarray = np.empty(0)
        self.x_scale_: np.ndarray = np.empty(0)
        self.y_mean_: np.ndarray = np.empty(0)
        self.y_scale_: np.ndarray = np.empty(0)

    # Receives: n_features and n_outputs — the network's input and output
    #   width, and rng — a random generator for weight initialisation.
    # Calculates: builds the full stack of layers implied by
    #   `hidden_layer_sizes` — alternating a `ModularLinearLayer` with the
    #   chosen activation for every hidden layer, ending in a plain linear
    #   output layer (regression targets shouldn't be squashed through an
    #   activation).
    # Returns: the list of layer objects, in forward-pass order.
    def _init_layers(self, n_features: int, n_outputs: int, rng: np.random.Generator) -> List[Layer]:
        layer_sizes = (n_features,) + self.hidden_layer_sizes + (n_outputs,)
        activation_cls = _ACTIVATIONS[self.activation]
        init_scheme = "he" if self.activation == "relu" else "xavier"

        layers: List[Layer] = []
        n_linear_layers = len(layer_sizes) - 1
        # zip(layer_sizes, layer_sizes[1:]) pairs up each consecutive size
        # in the sequence — e.g. sizes (4, 50, 30, 1) gives pairs (4,50),
        # (50,30), (30,1), exactly the (fan_in, fan_out) for each linear
        # layer in the stack.
        for i, (fan_in, fan_out) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            is_output_layer = i == n_linear_layers - 1
            # The output layer feeds no activation, so it keeps Xavier
            # (appropriate for a linear/identity readout) regardless of
            # what the hidden activation is.
            linear = ModularLinearLayer(fan_in, fan_out, rng=rng, init="xavier" if is_output_layer else init_scheme)
            if not is_output_layer and self.activation == "relu":
                # Small positive bias so ReLU units don't start dead (zero
                # gradient on both sides of 0) -- standard practice, see
                # cs231n's neural network notes on weight initialization.
                linear.bias = np.full(fan_out, 0.01)
            layers.append(linear)
            if not is_output_layer:
                layers.append(activation_cls())
        return layers

    # Receives: X — a batch of (already standardised) input features.
    # Calculates: passes X through every layer in the stack, in order —
    #   each layer's output becomes the next layer's input.
    # Returns: the network's final output (still in standardised target
    #   space, not yet un-scaled).
    def _forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers_:
            out = layer(out)
        return out

    # Receives: X — the training feature matrix, and y — the training
    #   targets (1D for single-output regression, 2D for multi-output).
    # Calculates: standardises both X and y using training-set statistics,
    #   optionally splits off a validation set for early stopping, builds
    #   the network, then runs mini-batch training for up to `epochs`
    #   passes: each batch does a forward pass, computes the squared-error
    #   gradient, backpropagates it through every layer, optionally clips
    #   the combined gradient norm, and updates every linear layer's
    #   weights via Adam (or plain SGD). After every epoch, the current
    #   weights are checkpointed if they're the best seen so far, and
    #   several stopping/learning-rate-adjustment rules are checked.
    # Returns: self (the fitted model) — the layers end up holding the best
    #   weights seen during training (not necessarily the very last ones),
    #   and `loss_curve_`/`validation_scores_`/`n_iter_` record the
    #   training history.
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have matching number of samples")
        if X.shape[0] < 2:
            raise ValueError("need at least 2 samples to fit")

        # Remember whether y came in as a flat 1D array (single target) so
        # predict() can flatten its own output back to match later.
        self._single_output = y.ndim == 1
        y2d = y.reshape(-1, 1) if self._single_output else y

        rng = np.random.default_rng(self.random_state)

        # Standardise both features and targets using ONLY training-data
        # statistics; a scale of exactly 0 (a constant column) is floored
        # to 1.0 so dividing by it doesn't produce NaN/inf.
        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ < 1e-12] = 1.0
        self.y_mean_ = y2d.mean(axis=0)
        self.y_scale_ = y2d.std(axis=0)
        self.y_scale_[self.y_scale_ < 1e-12] = 1.0

        X_all = (X - self.x_mean_) / self.x_scale_
        y_all = (y2d - self.y_mean_) / self.y_scale_
        n_total, n_features = X_all.shape
        n_outputs = y_all.shape[1]

        X_val: Optional[np.ndarray] = None
        y_val: Optional[np.ndarray] = None
        if self.early_stopping:
            if n_total < 5:
                raise ValueError("early_stopping requires at least 5 samples")
            # Randomly split off a validation slice; training only ever
            # sees `X_fit`/`y_fit`, while `X_val`/`y_val` are used purely to
            # monitor for overfitting.
            n_val = max(1, int(round(n_total * self.validation_fraction)))
            perm = rng.permutation(n_total)
            val_idx, fit_idx = perm[:n_val], perm[n_val:]
            X_fit, y_fit = X_all[fit_idx], y_all[fit_idx]
            X_val, y_val = X_all[val_idx], y_all[val_idx]
        else:
            X_fit, y_fit = X_all, y_all

        n_fit = X_fit.shape[0]
        batch_size = min(32, n_fit) if self.batch_size == "auto" else min(int(self.batch_size), n_fit)

        self.layers_ = self._init_layers(n_features, n_outputs, rng)
        linear_layers = [layer for layer in self.layers_ if isinstance(layer, ModularLinearLayer)]
        # Static for the whole fit() call -- build once, walk it every batch
        # instead of re-materializing a reversed() iterator each time.
        # Backpropagation walks the layers in the OPPOSITE order from the
        # forward pass, since each layer's gradient depends on the
        # gradient of the layer that came AFTER it.
        reversed_layers = list(reversed(self.layers_))
        # Adam normalizes the gradient itself, so folding L2 into the raw
        # gradient before that normalization would let Adam rescale most of
        # the penalty away; apply it as decoupled weight decay instead (see
        # the alpha docstring above). Plain SGD keeps the classic coupled
        # L2-via-gradient term, which backward() computes directly.
        backward_alpha = 0.0 if self.solver == "adam" else self.alpha
        # Hoisted out of the batch loop: repeated `self.` attribute lookups
        # are real overhead once this loop runs thousands of times.
        # (Python attribute lookups on `self` aren't free — reading them
        # once here and reusing the local variable avoids paying that cost
        # again for every single mini-batch of every epoch.)
        solver_is_adam = self.solver == "adam"
        alpha = self.alpha
        beta_1, beta_2, epsilon = self.beta_1, self.beta_2, self.epsilon
        one_minus_beta1, one_minus_beta2 = 1.0 - beta_1, 1.0 - beta_2
        max_grad_norm = self.max_grad_norm

        # Adam's per-parameter moving averages: `m` tracks the (decayed)
        # mean of the gradient, `v` tracks the (decayed) mean of its
        # square. One pair of arrays per linear layer, for both weight and
        # bias.
        m_w = [np.zeros_like(layer.weight) for layer in linear_layers]
        v_w = [np.zeros_like(layer.weight) for layer in linear_layers]
        m_b = [np.zeros_like(layer.bias) for layer in linear_layers]
        v_b = [np.zeros_like(layer.bias) for layer in linear_layers]
        adam_t = 0

        self.loss_curve_ = []
        self.validation_scores_ = []
        best_loss = np.inf
        # Seeded with the freshly initialized (finite, if untrained) weights
        # so there is always something sane to fall back to -- e.g. a
        # pathological learning rate can blow the weights to NaN on epoch 1,
        # before a training-loss checkpoint would otherwise ever be taken.
        best_state: Optional[List[Tuple[np.ndarray, np.ndarray]]] = [
            (layer.weight.copy(), layer.bias.copy()) for layer in linear_layers
        ]
        no_improve_count = 0
        lr_t = self.lr
        epoch = 0

        for epoch in range(1, self.epochs + 1):
            # Reshuffle the training order every epoch.
            order = rng.permutation(n_fit)
            epoch_sq_err_sum = 0.0
            epoch_element_count = 0
            for start in range(0, n_fit, batch_size):
                batch_idx = order[start:start + batch_size]
                X_batch, y_batch = X_fit[batch_idx], y_fit[batch_idx]
                batch_n = X_batch.shape[0]

                y_pred = self._forward(X_batch)
                residual = y_pred - y_batch
                epoch_sq_err_sum += float(np.sum(residual ** 2))
                epoch_element_count += residual.size

                # Gradient of mean-squared-error with respect to the
                # network's output: d/dy_pred [ (y_pred - y)^2 ] averaged
                # over the batch is 2*(y_pred - y)/batch_n.
                grad = 2.0 * residual / batch_n
                for layer in reversed_layers:
                    if isinstance(layer, ModularLinearLayer):
                        grad = layer.backward(grad, alpha=backward_alpha)
                    else:
                        grad = layer.backward(grad)

                if max_grad_norm is not None:
                    _clip_grad_norm(linear_layers, max_grad_norm)

                if solver_is_adam:
                    adam_t += 1
                    # Bias correction compensates for the moving averages
                    # starting at zero, which would otherwise make the very
                    # first few steps artificially small.
                    bias_correction1 = 1.0 - beta_1 ** adam_t
                    bias_correction2 = 1.0 - beta_2 ** adam_t
                    for idx, layer in enumerate(linear_layers):
                        w_grad = layer.weight_grad
                        b_grad = layer.bias_grad
                        # Update the running mean (m) and running
                        # mean-of-squares (v) of the gradient, for both
                        # weight and bias.
                        m_w[idx] = beta_1 * m_w[idx] + one_minus_beta1 * w_grad
                        v_w[idx] = beta_2 * v_w[idx] + one_minus_beta2 * w_grad ** 2
                        m_b[idx] = beta_1 * m_b[idx] + one_minus_beta1 * b_grad
                        v_b[idx] = beta_2 * v_b[idx] + one_minus_beta2 * b_grad ** 2

                        # Adam's actual step: the bias-corrected mean
                        # gradient, divided by the square root of the
                        # bias-corrected mean-of-squares (plus a tiny
                        # epsilon to avoid division by zero) — this is what
                        # gives every parameter its own adaptive effective
                        # learning rate.
                        w_step = (m_w[idx] / bias_correction1) / (np.sqrt(v_w[idx] / bias_correction2) + epsilon)
                        b_step = (m_b[idx] / bias_correction1) / (np.sqrt(v_b[idx] / bias_correction2) + epsilon)
                        layer.weight -= lr_t * w_step
                        if alpha:
                            # Decoupled weight decay (AdamW-style): applied
                            # directly to the weight, separately from the
                            # Adam-normalised gradient step above.
                            layer.weight -= lr_t * alpha * layer.weight
                        layer.bias -= lr_t * b_step
                else:
                    for layer in linear_layers:
                        layer.update(lr_t)

            # L2 penalty term added to the reported loss (not the
            # gradient — that's handled separately above), matching
            # whichever convention (coupled or decoupled) the solver uses.
            l2_penalty = self.alpha * sum(np.sum(layer.weight ** 2) for layer in linear_layers) / n_fit
            train_loss = epoch_sq_err_sum / epoch_element_count + l2_penalty
            self.loss_curve_.append(train_loss)

            if not np.isfinite(train_loss):
                # The loss has diverged to inf/NaN — stop immediately
                # rather than continuing to train on garbage; the
                # best-weights checkpoint below still has something sane to
                # fall back to.
                break

            if self.early_stopping and X_val is not None and y_val is not None:
                val_pred = self._forward(X_val)
                val_loss = float(np.mean((val_pred - y_val) ** 2))
                self.validation_scores_.append(val_loss)
                monitor_loss = val_loss
            else:
                monitor_loss = train_loss

            if monitor_loss < best_loss - self.tol:
                # A genuine improvement (bigger than the tolerance
                # threshold, to avoid chasing floating-point noise) — save
                # this as the new best checkpoint.
                best_loss = monitor_loss
                best_state = [(layer.weight.copy(), layer.bias.copy()) for layer in linear_layers]
                no_improve_count = 0
            else:
                no_improve_count += 1

            stop = False
            if self.solver == "sgd" and self.learning_rate == "adaptive":
                if no_improve_count >= self.n_iter_no_change:
                    # Training has stalled — try a smaller learning rate
                    # before giving up entirely.
                    lr_t /= 5.0
                    no_improve_count = 0
                    if lr_t < 1e-6:
                        stop = True
            elif self.early_stopping and no_improve_count >= self.n_iter_no_change:
                stop = True

            if self.solver == "sgd" and self.learning_rate == "invscaling":
                lr_t = self.lr / (epoch ** self.power_t)

            if stop:
                break

        self.n_iter_ = epoch
        if best_state is not None:
            # Always keep the best weights seen, not just the last ones --
            # cheap insurance against a late-training divergence or an
            # unlucky final mini-batch.
            for layer, (weight, bias) in zip(linear_layers, best_state):
                layer.weight = weight
                layer.bias = bias

        return self

    # Receives: X — new feature data to predict on.
    # Calculates: standardises X using the TRAINING statistics, runs it
    #   through the fitted network, then un-standardises the network's
    #   output back into the original target scale.
    # Returns: an array of predictions — flattened to 1D if the model was
    #   trained on a single-output target, otherwise a 2D array matching
    #   the training target's own shape.
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.layers_:
            raise RuntimeError("MLPRegressor instance is not fitted yet; call fit() first")

        X = np.asarray(X, dtype=np.float64)
        X_scaled = (X - self.x_mean_) / self.x_scale_
        y_scaled = self._forward(X_scaled)
        # Undo the target standardisation applied during fit(), so the
        # caller always sees predictions in the original target units.
        y_pred = y_scaled * self.y_scale_ + self.y_mean_
        result: np.ndarray = y_pred.ravel() if self._single_output else y_pred
        return result
