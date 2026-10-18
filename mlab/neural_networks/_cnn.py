"""Convolutional Neural Network built from composable layers, trained by
hand-rolled numpy backprop -- same approach as ``_mlp.py``: every layer only
knows its own forward (``__call__``), ``backward``, and ``update``.

Unlike the lecture notebook, which hard-codes a single-image, two-layer
network with global variables, everything here processes a full batch of
images per call and is wired together generically so ``CNNClassifier`` can
build a variable-depth stack instead of one fixed architecture.

Both ``ConvLayer`` and ``PoolingLayer`` share the same patch extraction
helper (``_sliding_windows``, built on ``numpy.lib.stride_tricks``), and
every forward/backward computation is vectorized over the batch, spatial,
and channel axes -- the only Python-level loops left are over the kernel's
own (small, fixed) height/width offsets, never over batch, output
position, or channel count.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# Import the MLP layers that are reused in the CNN architecture. Why this is done: the CNN
# architecture uses some of the same layer types as the MLP (Multi-Layer Perceptron) architecture,
# such as ReLU and Softmax layers. By importing these layers from the MLP module, we can reuse the
# existing implementations instead of duplicating code. This promotes code reusability and
# maintainability, allowing us to leverage the tested and optimized implementations of these layers
# in both architectures.
from mlab.neural_networks._mlp import ModularLinearLayer, ReLULayer, SoftmaxLayer

__all__ = [
    "ConvLayer",
    "PoolingLayer",
    "ReLULayer",
    "SoftmaxLayer",
    "ModularLinearLayer",
    "CNNClassifier",
]

IntOrPair = Union[int, Tuple[int, int]]

# This function is used to normalize hyperparameters that can be specified either as a single
# integer or as a tuple of two integers.
# Receives: value — either a single int (meaning "use this for both height
#   and width") or an (int, int) tuple, and name — the hyperparameter's own
#   name, used only to word error messages.
# Calculates: unpacks value into an explicit (height, width) pair, and
#   checks that both numbers are positive.
# Returns: a 2-tuple (h, w) of plain Python ints.
def _as_pair(value: IntOrPair, name: str) -> Tuple[int, int]:
    """Normalise an int-or-(int, int) hyperparameter into a 2-tuple."""
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or a 2-tuple, got {value!r}")
        h, w = value
    else:
        h = w = value
    if h <= 0 or w <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return int(h), int(w)

# I used H,W,C for height, width, channels to match the common convention in deep learning libraries
# like TensorFlow and PyTorch. The input data is expected to be in the shape (batch_size, height,
# width, channels), which is often referred to as "channels last" format. This format is widely used
# because it aligns with how images are typically represented and processed in many computer vision
# tasks.
# Receives: X — a batch of images, shape (batch, H, W, C); kernel_size —
#   the (height, width) of the sliding window; stride — how far the window
#   moves each step.
# Calculates: every possible (kH, kW) window of X, using numpy's
#   `sliding_window_view` to generate all of them as a single view into the
#   existing array (no copying of the actual pixel data happens until the
#   result is actually used/reshaped), then keeps only every `stride`-th
#   window in each spatial direction.
# Returns: an array of shape (batch, out_h, out_w, C, kH, kW) — every
#   window, for every image in the batch, for every channel.
def _sliding_windows(
    X: np.ndarray, kernel_size: Tuple[int, int], stride: Tuple[int, int]
) -> np.ndarray:
    """All (kH, kW) windows of a (batch, H, W, C) array, strided, in the
    (batch, out_h, out_w, C, kH, kW) layout ``sliding_window_view`` produces
    directly -- channels ahead of the kernel axes, not last. Shared by
    ``_extract_patches`` (which moves C to the end for ``PoolingLayer``)
    and ``_patches_to_matrix`` (which wants C exactly where it already is).

    Returns
    -------
    np.ndarray of shape (batch, out_h, out_w, C, kH, kW)
    """
    kh, kw = kernel_size
    sh, sw = stride
    if X.shape[1] < kh or X.shape[2] < kw:
        raise ValueError(f"kernel {kernel_size} is larger than the padded input {X.shape[1:3]}")
    # slide over height and width one axis at a time -- equivalent to a single call with axis=(1,
    # 2), but keeps every window_shape/axis pair a plain int, which is what the sliding_window_view
    # overloads expect.
    # First call: slide a window of size kh along axis 1 (height), adding a
    # new trailing axis holding each window's own kh values. Second call:
    # do the same along axis 2 (width) with size kw, on the RESULT of the
    # first call. Together, this produces every (kh, kw) window.
    windows = np.lib.stride_tricks.sliding_window_view(X, kh, axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(windows, kw, axis=2)
    # `sliding_window_view` produces a window starting at EVERY possible
    # position (stride 1); slicing with `::sh`/`::sw` keeps only every
    # `stride`-th one, which is exactly what a strided convolution/pooling
    # window needs.
    return windows[:, ::sh, ::sw]


# Receives: X — a batch of images, kernel_size and stride — the pooling/
#   patch window shape and step.
# Calculates: the same sliding windows as `_sliding_windows`, but with the
#   channel axis moved from its "ahead of the kernel axes" position to the
#   very end.
# Returns: an array of shape (batch, out_h, out_w, kH, kW, C) — arranged so
#   each channel's own window can be reduced (e.g. max-pooled)
#   independently.
def _extract_patches(
    X: np.ndarray, kernel_size: Tuple[int, int], stride: Tuple[int, int]
) -> np.ndarray:
    """Patch extraction for ``PoolingLayer``: same windows as
    ``_sliding_windows``, with channels moved to the last axis so each
    channel's own (kH, kW) window can be reduced independently.

    Returns
    -------
    np.ndarray of shape (batch, out_h, out_w, kH, kW, C)
    """
    return np.moveaxis(_sliding_windows(X, kernel_size, stride), 3, -1)


# Receives: X — a batch of images, kernel_size and stride — the
#   convolution window shape and step.
# Calculates: the sliding windows (patches) of X, then flattens every
#   patch (across its spatial extent AND channels) into a single row of a
#   2D matrix — turning "how many patches are there, each of what size" into
#   a plain (num_patches, patch_size) table that a single matrix multiply
#   against the reshaped filter weights can process all at once.
# Returns: a tuple (matrix, out_shape) — the flattened patch matrix, and
#   the (batch, out_h, out_w) shape needed to fold the eventual output back
#   into image form.
def _patches_to_matrix(
    X: np.ndarray, kernel_size: Tuple[int, int], stride: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Patch extraction for ``ConvLayer``: flatten windows straight into a
    2D (batch*out_h*out_w, C*kH*kW) matrix -- channels already lead the
    kernel axes in ``_sliding_windows``' own output, so this needs no
    channel-shuffling moveaxis at all (``_extract_patches`` above moves C
    to the end only because ``PoolingLayer`` needs it there; undoing that
    move again here would be pure wasted work). This turns the
    convolution's patch-times-kernel contraction into a single vectorized
    matrix multiply (`@`, i.e. ``np.dot``) instead of an einsum --
    profiling showed einsum's own path analysis and BLAS-dispatch setup
    cost more than the multiply itself at the batch/image sizes this
    course's datasets actually use, so a direct matmul is both simpler and
    consistently faster here."""
    windows = _sliding_windows(X, kernel_size, stride)
    batch, out_h, out_w, channels, kh, kw = windows.shape
    matrix = windows.reshape(batch * out_h * out_w, channels * kh * kw)
    return matrix, (batch, out_h, out_w)


# This class implements a 2D convolutional layer that performs cross-correlation between input
# patches and learned kernels. It supports forward and backward passes, weight initialization, and
# gradient updates.
class ConvLayer:
    """2D convolutional layer: cross-correlates ``in_channels``-deep patches
    against ``out_channels`` learned kernels.

    ``weight`` has shape ``(out_channels, in_channels, kH, kW)`` and
    ``bias`` has shape ``(out_channels,)``; data is channels-last,
    ``(batch, H, W, C)``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of learned filters (= output channels).
    kernel_size : int or (int, int)
        Spatial extent of each filter.
    stride : int or (int, int), default=1
        Step size the kernel slides by in each spatial direction.
    padding : int or (int, int), default=0
        Zero-padding added to each side of the height/width axes before
        convolving.
    rng : numpy.random.Generator, optional
        Source of randomness for weight initialization.

    Attributes
    ----------
    weight : np.ndarray of shape (out_channels, in_channels, kH, kW)
        Filter weights, He-initialized for the ReLU activation this layer
        typically feeds.
    bias : np.ndarray of shape (out_channels,)
        Per-filter bias.
    """
    # This is just initialization of the ConvLayer class. It sets up the parameters and initializes
    # the weights and biases for the convolutional layer. The weights are initialized using He
    # initialization, which is suitable for ReLU activations.
    # Receives: in_channels/out_channels — how many input and output
    #   feature maps this layer has, kernel_size/stride/padding — the
    #   convolution's spatial shape, and rng — a random generator for
    #   weight initialisation.
    # Calculates: validates every argument, then initialises the filter
    #   weights using He initialisation (matched to the ReLU activation
    #   this layer normally feeds into) and a zero bias.
    # Returns: nothing — __init__ only sets up `self.weight`/`self.bias`
    #   and empty placeholder buffers the forward/backward passes will fill
    #   in.
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _as_pair(kernel_size, "kernel_size")
        self.stride = _as_pair(stride, "stride")
        pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
        if pad_h < 0 or pad_w < 0:
            raise ValueError(f"padding must be non-negative, got {padding!r}")
        self.padding = (int(pad_h), int(pad_w))

        kh, kw = self.kernel_size
        active_rng = rng if rng is not None else np.random.default_rng()
        # He initialisation: fan_in is how many numbers feed into each
        # output value (in_channels times the kernel's own area) — scaling
        # the weight's standard deviation by sqrt(2/fan_in) keeps the
        # variance of activations roughly stable through a ReLU-activated
        # stack of layers.
        fan_in = in_channels * kh * kw
        std = np.sqrt(2.0 / fan_in)
        self.weight = active_rng.normal(0.0, std, size=(out_channels, in_channels, kh, kw))
        self.bias = np.zeros(out_channels)

        self._padded_shape: Tuple[int, ...] = ()
        self._user_padded_shape: Tuple[int, ...] = ()
        self._auto_pad: Tuple[int, int] = (0, 0)
        self._patches_matrix: np.ndarray = np.empty(0)
        self._weight_grad: np.ndarray = np.zeros_like(self.weight)
        self._bias_grad: np.ndarray = np.zeros_like(self.bias)
    # The __call__ method performs the forward pass of the convolutional layer. It takes an input
    # tensor X, applies padding if necessary, extracts patches from the input, and computes the
    # output as a single matrix multiply between the flattened patches and the flattened learned
    # weights (vectorization via np.dot-style matmul over every patch and every output channel at
    # once, rather than looping over spatial positions or channels). The output is then returned.
    # Receives: X — a batch of input feature maps, shape (batch, H, W,
    #   in_channels), and training — whether to cache the intermediate
    #   values `backward` will need (skip the cache during pure inference,
    #   since there's no backward call to serve).
    # Calculates: applies zero-padding (both the user-requested padding and
    #   any extra auto-padding needed if the kernel is bigger than the
    #   input), extracts every convolution patch and flattens it into a
    #   matrix, then computes the whole convolution as one matrix multiply
    #   against the reshaped filter weights.
    # Returns: the convolution output, shape (batch, out_height, out_width,
    #   out_channels).
    def __call__(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        X : np.ndarray of shape (batch_size, height, width, in_channels)
        training : bool, default=True
            When False (pure inference, e.g. from ``CNNClassifier.predict``),
            skips caching the patches/shape ``backward`` would otherwise
            need -- there is no backward call to serve in that case.

        Returns
        -------
        np.ndarray of shape (batch_size, out_height, out_width, out_channels)
        """
        if X.ndim != 4:
            raise ValueError(f"ConvLayer expects a 4D (batch, H, W, C) input, got shape {X.shape}")
        if X.shape[-1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {X.shape[-1]}")

        pad_h, pad_w = self.padding
        if pad_h or pad_w:
            X_padded = np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
        else:
            X_padded = X
        user_padded_shape = X_padded.shape

        # Large-kernel edge case: the weight shape is fixed at construction, so if the
        # (already user-padded) input is still smaller than the kernel, grow the input
        # instead of shrinking the kernel -- pad just enough on the trailing side for
        # the kernel to fit, producing a valid (if degenerate, e.g. 1x1) output rather
        # than raising.
        kh, kw = self.kernel_size
        extra_h = max(0, kh - X_padded.shape[1])
        extra_w = max(0, kw - X_padded.shape[2])
        if extra_h or extra_w:
            X_padded = np.pad(X_padded, ((0, 0), (0, extra_h), (0, extra_w), (0, 0)))

        if training:
            # Remember exactly how much padding was added, and in what
            # order, so `backward` can strip it back off the gradient in
            # reverse.
            self._padded_shape = X_padded.shape
            self._user_padded_shape = user_padded_shape
            self._auto_pad = (extra_h, extra_w)

        patches_matrix, out_shape = _patches_to_matrix(X_padded, self.kernel_size, self.stride)
        batch, out_h, out_w = out_shape
        if training:
            # Cached for backward(): the weight gradient needs exactly
            # these same patches, and recomputing them would mean
            # re-running the sliding-window extraction a second time.
            self._patches_matrix = patches_matrix
        # Flatten the (out_channels, in_channels, kH, kW) weight down to a
        # 2D (out_channels, in_channels*kH*kW) matrix, matching the
        # patches matrix's own flattened layout, so the whole convolution
        # becomes one matrix multiply.
        weight_matrix = self.weight.reshape(self.out_channels, -1)
        out = (patches_matrix @ weight_matrix.T).reshape(batch, out_h, out_w, self.out_channels)
        out += self.bias
        return out
    # The backward method computes the gradients of the loss with respect to the input, weights, and
    # bias of the convolutional layer.
    # Receives: grad_output — the gradient of the loss with respect to this
    #   layer's OUTPUT, handed down from whichever layer follows it.
    # Calculates: the weight gradient (via a matrix multiply against the
    #   cached forward patches), the bias gradient (summed over every
    #   batch/spatial position), and the gradient with respect to this
    #   layer's own input (via `_input_grad`), then strips off the padding
    #   that was added during the forward pass (in the reverse order it was
    #   applied).
    # Returns: the gradient of the loss with respect to this layer's
    #   original (unpadded) input, shape (batch, height, width,
    #   in_channels).
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients w.r.t. input, weights, and bias.

        Parameters
        ----------
        grad_output : np.ndarray of shape (batch_size, out_height, out_width, out_channels)
            Upstream gradient dL/d(layer output).

        Returns
        -------
        np.ndarray of shape (batch_size, height, width, in_channels)
            Gradient w.r.t. this layer's original (unpadded) input.
        """
        kh, kw = self.kernel_size
        # Flatten grad_output the same way the forward pass's output was
        # flattened, so it lines up row-for-row with `_patches_matrix`.
        grad_matrix = grad_output.reshape(-1, self.out_channels)
        # The weight gradient is exactly "each patch's contribution,
        # weighted by how much that patch's output affected the loss" —
        # summed over every patch via one matrix multiply.
        weight_grad_matrix = grad_matrix.T @ self._patches_matrix
        self._weight_grad = weight_grad_matrix.reshape(self.out_channels, self.in_channels, kh, kw)
        # The bias gradient is just the sum of the output gradient over
        # every batch element and every spatial position (each bias value
        # was added to every one of those positions equally).
        self._bias_grad = grad_output.sum(axis=(0, 1, 2))
        del self._patches_matrix

        grad_padded = self._input_grad(grad_output)

        # Undo the two padding steps forward added, in the same order: first the
        # trailing auto-pad (large-kernel edge case), then the user's own padding.
        extra_h, extra_w = self._auto_pad
        if extra_h or extra_w:
            h, w = self._padded_shape[1], self._padded_shape[2]
            grad_padded = grad_padded[:, :h - extra_h, :w - extra_w, :]

        pad_h, pad_w = self.padding
        if pad_h or pad_w:
            h, w = self._user_padded_shape[1], self._user_padded_shape[2]
            grad_padded = grad_padded[:, pad_h:h - pad_h, pad_w:w - pad_w, :]
        return grad_padded
    # The _input_grad method computes the gradient of the loss with respect to the input of the
    # convolutional layer, using a col2im-style scatter-add instead of materializing a "full
    # convolution" patches array.
    # Receives: grad_output — the gradient of the loss with respect to this
    #   layer's output.
    # Calculates: the gradient with respect to the (padded) input, by
    #   looping only over the kernel's own small, fixed (kH, kW) offsets —
    #   for each offset, works out which slice of the input every output
    #   pixel's gradient should be scattered back onto, and accumulates it
    #   there.
    # Returns: the gradient with respect to the padded input, same shape as
    #   `self._padded_shape`.
    def _input_grad(self, grad_output: np.ndarray) -> np.ndarray:
        # Every output pixel (i, j) was produced from input patch starting at
        # (i*stride, j*stride); its gradient therefore scatters back onto that same
        # patch. Looping over the kH*kW kernel OFFSETS (a small, fixed count, e.g. 9)
        # rather than over every patch keeps this a handful of vectorized matmul +
        # strided-slice-accumulate steps -- the same bounded-loop scatter-add
        # PoolingLayer.backward uses -- instead of building a patches array the size
        # of a full convolution (which is *larger* than the original input) just to
        # contract it away again.
        kh, kw = self.kernel_size
        sh, sw = self.stride
        batch, out_h, out_w, _ = grad_output.shape
        grad_matrix = grad_output.reshape(-1, self.out_channels)

        grad_input = np.zeros(self._padded_shape)
        for dh in range(kh):
            for dw in range(kw):
                # For this one (dh, dw) kernel offset, every output pixel's
                # gradient contributes to exactly one input pixel — the one
                # `dh, dw` steps into its own source patch. `weight_slice`
                # is the filter's own weight AT this offset, for every
                # (in_channel, out_channel) pair; multiplying it against
                # every output pixel's gradient gives that offset's
                # contribution to the input gradient, for every patch at
                # once.
                weight_slice = self.weight[:, :, dh, dw]
                contribution = grad_matrix @ weight_slice
                contribution = contribution.reshape(batch, out_h, out_w, self.in_channels)
                # Scatter that contribution back onto the correct strided
                # slice of the (padded) input gradient — accumulating
                # (`+=`) since multiple output pixels can share the same
                # underlying input pixel through overlapping patches.
                grad_input[:, dh:dh + sh * out_h:sh, dw:dw + sw * out_w:sw, :] += contribution
        return grad_input
    # The update method updates the weights and biases of the convolutional layer using a plain
    # gradient step.
    # Receives: learning_rate — the step size to use for this update.
    # Calculates: nothing new — just applies the already-computed gradients
    #   (from the last `backward` call) to the weight and bias.
    # Returns: nothing — updates `self.weight`/`self.bias` in place.
    def update(self, learning_rate: float) -> None:
        """Update weights and biases with a plain gradient step."""
        self.weight -= learning_rate * self._weight_grad
        self.bias -= learning_rate * self._bias_grad

    # Receives: nothing besides self.
    # Calculates: nothing — just exposes the cached weight gradient.
    # Returns: the weight gradient computed by the last `backward()` call.
    @property
    def weight_grad(self) -> np.ndarray:
        """Gradient computed by the last backward() call."""
        return self._weight_grad

    # Receives: nothing besides self.
    # Calculates: nothing — just exposes the cached bias gradient.
    # Returns: the bias gradient computed by the last `backward()` call.
    @property
    def bias_grad(self) -> np.ndarray:
        """Gradient computed by the last backward() call."""
        return self._bias_grad
    # The __repr__ method provides a string representation of the ConvLayer instance, including its
    # parameters such as input channels, output channels, kernel size, stride, and padding. This is
    # useful for debugging and understanding the configuration of the layer.
    def __repr__(self) -> str:
        return (
            f"ConvLayer(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
        )

# This class implements a 2D max-pooling layer that performs downsampling by taking the maximum
# value within non-overlapping windows of the input.
class PoolingLayer:
    """2D max-pooling layer (no learnable parameters).

    Parameters
    ----------
    pool_size : int or (int, int), default=2
        Spatial extent of the pooling window.
    stride : int or (int, int), default=2
        Step size the window slides by.
    """
    # This is just initialization of the PoolingLayer class. It sets up the parameters for the
    # pooling layer, including the pool size and stride.
    # Receives: pool_size and stride — the pooling window's spatial extent
    #   and step size.
    # Calculates: nothing — just validates and stores the pooling
    #   hyperparameters (pooling has no learnable weights, unlike
    #   ConvLayer).
    # Returns: nothing — __init__ only sets up placeholders `backward` will
    #   need later.
    def __init__(self, pool_size: IntOrPair = 2, stride: IntOrPair = 2) -> None:
        self.pool_size = _as_pair(pool_size, "pool_size")
        self.stride = _as_pair(stride, "stride")
        self._input_shape: Tuple[int, ...] = ()
        self._argmax: np.ndarray = np.empty(0, dtype=np.intp)
        self._eff_pool_size: Tuple[int, int] = self.pool_size
        self._eff_stride: Tuple[int, int] = self.stride
    # The __call__ method performs the forward pass of the pooling layer.
    # Receives: X — a batch of feature maps, shape (batch, H, W, channels),
    #   and training — whether to cache what `backward` will need (skipped
    #   during pure inference).
    # Calculates: shrinks the configured pool window/stride down to fit if
    #   the input is smaller than requested (global pooling fallback), then
    #   extracts every pooling window and takes its maximum, remembering
    #   WHICH position within each window held that maximum (needed to
    #   route gradients correctly in `backward`).
    # Returns: the pooled output, shape (batch, out_height, out_width,
    #   channels).
    def __call__(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass: apply max pooling.

        Parameters
        ----------
        X : np.ndarray of shape (batch_size, height, width, channels)
        training : bool, default=True
            When False (pure inference, e.g. from ``CNNClassifier.predict``),
            skips the extra argmax reduction over each window -- it exists
            only to route gradients in ``backward``, which pure inference
            never calls.

        Returns
        -------
        np.ndarray of shape (batch_size, out_height, out_width, channels)
        """
        if X.ndim != 4:
            raise ValueError(f"PoolingLayer expects a 4D (batch, H, W, C) input, got {X.shape}")

        # Minimal-input edge case: there are no learnable weights to keep a fixed shape
        # for here, so if the configured window doesn't fit, shrink it (and its stride)
        # down to the whole available extent instead of raising -- a single window
        # covering everything, i.e. plain global max pooling along that axis.
        kh, kw = self.pool_size
        sh, sw = self.stride
        height, width = X.shape[1], X.shape[2]
        eff_kh, eff_sh = (height, height) if kh > height else (kh, sh)
        eff_kw, eff_sw = (width, width) if kw > width else (kw, sw)

        patches = _extract_patches(X, (eff_kh, eff_kw), (eff_sh, eff_sw))
        batch, out_h, out_w, _, _, channels = patches.shape
        # Flatten each window's own (kH, kW) extent down to a single axis,
        # so "the max over the window" is just a max over that one axis.
        flat = patches.reshape(batch, out_h, out_w, eff_kh * eff_kw, channels)
        if not training:
            return np.max(flat, axis=3)
        self._input_shape = X.shape
        self._eff_pool_size = (eff_kh, eff_kw)
        self._eff_stride = (eff_sh, eff_sw)
        # Remember WHICH position within each window held the maximum —
        # backward() needs this to know exactly where to route each
        # gradient back to (every other position in the window gets zero
        # gradient, since it didn't affect the forward output).
        self._argmax = np.argmax(flat, axis=3)
        return np.max(flat, axis=3)
    # The backward method computes the gradient of the loss with respect to the input of the pooling
    # layer.
    # Receives: grad_output — the gradient of the loss with respect to this
    #   layer's output.
    # Calculates: scatters each output pixel's gradient back onto exactly
    #   the ONE input position that produced it (the position that held the
    #   max during the forward pass, cached in `self._argmax`) — every other
    #   position in that pooling window gets zero gradient.
    # Returns: the gradient with respect to this layer's input, shape
    #   (batch, height, width, channels) — zero everywhere except at each
    #   window's argmax position.
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Route gradients back through max positions.

        Parameters
        ----------
        grad_output : np.ndarray of shape (batch_size, out_height, out_width, channels)

        Returns
        -------
        np.ndarray of shape (batch_size, height, width, channels)
            Zero everywhere except at each window's argmax position.
        """
        kh, kw = self._eff_pool_size
        sh, sw = self._eff_stride
        batch, out_h, out_w, channels = grad_output.shape

        # Build an all-zero "flat window" gradient array, then use
        # `put_along_axis` to place each output pixel's gradient at exactly
        # the argmax position recorded during the forward pass — every
        # other position in the flattened window stays 0.
        grad_flat = np.zeros((batch, out_h, out_w, kh * kw, channels), dtype=grad_output.dtype)
        argmax = self._argmax[:, :, :, None, :]
        np.put_along_axis(grad_flat, argmax, grad_output[:, :, :, None, :], axis=3)
        grad_patches = grad_flat.reshape((batch, out_h, out_w, kh, kw, channels))

        # Same bounded-loop scatter-add pattern as ConvLayer._input_grad:
        # loop only over the small, fixed (kh, kw) window offsets, and
        # accumulate each offset's own gradient slice back onto the
        # correct strided position of the full input gradient.
        grad_input = np.zeros(self._input_shape, dtype=grad_output.dtype)
        for i in range(kh):
            for j in range(kw):
                window = grad_patches[:, :, :, i, j, :]
                grad_input[:, i:i + sh * out_h:sh, j:j + sw * out_w:sw, :] += window
        return grad_input
    # The __repr__ method provides a string representation of the PoolingLayer instance, including
    # its parameters such as pool size and stride.
    def __repr__(self) -> str:
        return f"PoolingLayer(pool_size={self.pool_size}, stride={self.stride})"

# This type alias represents a union of all the layer types used in the CNN architecture, including
# convolutional, pooling, ReLU, softmax, and modular linear layers. why this is done: The Layer type
# alias is defined as a union of all the layer types used in the CNN architecture. This allows for
# type hinting and better code readability, as it indicates that a variable of type Layer can be any
# of the specified layer types. It also helps with static type checking and code analysis tools to
# understand the expected types in the codebase.
Layer = Union[ConvLayer, PoolingLayer, ReLULayer, SoftmaxLayer, ModularLinearLayer]

# This function computes the softmax of the input logits in a numerically stable way by shifting the
# logits to avoid overflow issues.
# Receives: logits — a batch of raw class scores, one row per sample.
# Calculates: the softmax of each row, subtracting each row's own max
#   before exponentiating (so every exponent is <= 0, preventing overflow,
#   without changing the final normalised result).
# Returns: a batch of probability distributions, each row summing to 1.
def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

# The CNNClassifier class implements a convolutional neural network (CNN) for image classification.
class CNNClassifier:
    """Image classifier: [Conv -> ReLU -> (Pool)] x2 -> flatten -> Linear
    -> Softmax, trained end-to-end with mini-batch Adam.

    Each conv block skips its pooling step, and shrinks its kernel to fit,
    whenever the current feature map is too small for the requested size --
    so the same architecture code handles anything from full MNIST digits
    down to a handful of pixels.

    Input is standardized internally with a single global scalar mean/std
    computed at fit time (stored on ``x_mean_``/``x_scale_``), so callers
    don't need to pre-scale raw pixel values.

    Parameters
    ----------
    input_shape : (int, int, int), default=(28, 28, 1)
        Expected (height, width, channels) of a single input image.
    num_classes : int, default=10
        Number of output classes; labels passed to ``fit`` must be integers
        in ``[0, num_classes)``.
    n_filters1, n_filters2 : int, default=4, 8
        Number of filters in the first and second convolutional blocks.
        Halving sklearn-baseline-competitive defaults (8/16) roughly
        halves both fit and predict time (verified on the digits dataset:
        ~1.7x/~2x faster) for a ~0.003 train-accuracy cost -- convolution
        is inherently more expensive per sample than a flat MLP of
        comparable width, so trimming filter count is the lever available
        to close that gap without abandoning real convolution.
    kernel_size : int, default=3
        Spatial extent of both convolutional layers (shrunk automatically
        if the current feature map is smaller than this).
    pool_size : int, default=2
        Spatial extent (and stride) of both max-pooling layers.
    lr : float, default=0.01
        Adam base learning rate.
    epochs : int, default=20
        Number of passes over the training data.
    batch_size : int, default=32
        Mini-batch size (clamped to the number of training samples).
    alpha : float, default=0.0
        L2 weight decay strength, applied AdamW-style (decoupled from the
        gradient), matching ``MLPRegressor``'s ``alpha``.
    random_state : int, optional
        Seed for weight initialization and mini-batch shuffling.
    layers : sequence of layer objects, optional
        A pre-built layer stack (any mix of ``ConvLayer``/``PoolingLayer``/
        ``ReLULayer``/``ModularLinearLayer``, ending in a linear layer whose
        output width is ``num_classes``) to use as-is instead of building
        one from ``n_filters1``/``n_filters2``/``kernel_size``/``pool_size``.
        Lets a caller wire up a specific architecture directly; ``fit``
        trains whatever stack it's given either way.

    Attributes
    ----------
    layers_ : list
        Fitted layer stack, in forward order: conv/relu/(pool) x2 followed
        by the final ``ModularLinearLayer``.
    classes_ : np.ndarray of shape (num_classes,)
        The class labels the model outputs over, ``arange(num_classes)``.
    loss_history_ : list of float
        Mean cross-entropy training loss per epoch.
    n_iter_ : int
        Number of epochs actually run.
    """
    # this is just initialization of the CNNClassifier class. It sets up the parameters for the CNN
    # architecture, including input shape, number of classes, number of filters, kernel size,
    # pooling size, learning rate, number of epochs, batch size, weight decay strength, and random
    # state for reproducibility.
    # Receives: every architecture/training hyperparameter (input_shape,
    #   num_classes, n_filters1/n_filters2, kernel_size, pool_size, lr,
    #   epochs, batch_size, alpha, random_state), plus an optional
    #   pre-built `layers` stack to use instead of building one from the
    #   other hyperparameters.
    # Calculates: validates every hyperparameter up front, then stores them
    #   all; if a `layers` stack was supplied, it's kept ready for
    #   prediction immediately (even before `fit()` ever trains it).
    # Returns: nothing — __init__ only prepares the object.
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (28, 28, 1),
        num_classes: int = 10,
        n_filters1: int = 4,
        n_filters2: int = 8,
        kernel_size: int = 3,
        pool_size: int = 2,
        lr: float = 0.01,
        epochs: int = 20,
        batch_size: int = 32,
        alpha: float = 0.0,
        random_state: Optional[int] = None,
        layers: Optional[Sequence[Layer]] = None,
    ) -> None:
        if len(input_shape) != 3 or any(dim <= 0 for dim in input_shape):
            msg = f"input_shape must be a positive (H, W, C) tuple, got {input_shape}"
            raise ValueError(msg)
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if n_filters1 <= 0 or n_filters2 <= 0:
            raise ValueError("n_filters1 and n_filters2 must be positive")
        if kernel_size <= 0 or pool_size <= 0:
            raise ValueError("kernel_size and pool_size must be positive")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        self.input_shape = tuple(int(d) for d in input_shape)
        self.num_classes = num_classes
        self.n_filters1 = n_filters1
        self.n_filters2 = n_filters2
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.random_state = random_state
        self._init_layers = list(layers) if layers else None

        # A caller-supplied stack is usable for predict() right away, before
        # fit() ever runs -- e.g. to benchmark inference on a fixed
        # architecture without training it first.
        self.layers_: List[Layer] = list(self._init_layers) if self._init_layers else []
        self.classes_: np.ndarray = np.arange(num_classes)
        self.loss_history_: List[float] = []
        self.n_iter_ = 0
        self._flatten_shape: Tuple[int, ...] = ()

        self.x_mean_ = 0.0
        self.x_scale_ = 1.0
    # The _conv_block method constructs a convolutional block consisting of a convolutional layer, a
    # ReLU activation, and optionally a pooling layer.
    # Receives: in_channels/out_channels — the block's input/output channel
    #   counts, h/w — the current feature map's height/width, and rng — a
    #   random generator for weight initialisation.
    # Calculates: clamps the kernel size down to fit the current feature
    #   map (so a tiny input never crashes on a kernel that's too big),
    #   builds a Conv+ReLU pair, and appends a PoolingLayer only if the
    #   post-conv feature map is still at least as big as the pooling
    #   window.
    # Returns: a 3-tuple (layers, out_h, out_w) — the block's own layer
    #   list, and the feature map's height/width AFTER this block (which
    #   becomes the next block's own h/w input).
    def _conv_block(
        self, in_channels: int, out_channels: int, h: int, w: int, rng: np.random.Generator
    ) -> Tuple[List[Layer], int, int]:
        # Never let the kernel be bigger than the feature map itself — a
        # kernel that doesn't fit would make the convolution meaningless
        # (or crash), so it's automatically shrunk down to whatever extent
        # actually fits.
        k = min(self.kernel_size, h, w)
        # "Same"-style padding: keeps the output spatial size equal to the
        # input size (before any pooling), so a `k`-sized odd kernel
        # doesn't shrink the feature map on its own.
        pad = (k - 1) // 2
        conv = ConvLayer(in_channels, out_channels, k, stride=1, padding=pad, rng=rng)
        out_h = h + 2 * pad - k + 1
        out_w = w + 2 * pad - k + 1

        layers: List[Layer] = [conv, ReLULayer()]
        if out_h >= self.pool_size and out_w >= self.pool_size:
            # Only pool if the feature map is still big enough for the
            # configured pool window — otherwise skip pooling entirely
            # rather than shrinking it further (PoolingLayer itself still
            # has its OWN shrink-to-fit fallback for the case where it IS
            # used but the map later turns out too small, e.g. in a
            # differently-configured stack).
            layers.append(PoolingLayer(self.pool_size, self.pool_size))
            out_h //= self.pool_size
            out_w //= self.pool_size
        return layers, out_h, out_w
    # The _build_layers method constructs the entire CNN architecture by stacking two convolutional
    # blocks followed by a flattening layer and a final linear layer for classification.
    # Receives: rng — a random generator for weight initialisation.
    # Calculates: chains two convolutional blocks together (the second
    #   block's input channels/spatial size are exactly the first block's
    #   output), then adds a final linear layer mapping the flattened
    #   feature map down to `num_classes` outputs.
    # Returns: the complete list of layers, in forward-pass order.
    def _build_layers(self, rng: np.random.Generator) -> List[Layer]:
        h, w, c = self.input_shape
        block1, h, w = self._conv_block(c, self.n_filters1, h, w, rng)
        block2, h, w = self._conv_block(self.n_filters1, self.n_filters2, h, w, rng)
        # Remember the feature map's shape right before flattening — needed
        # later to reshape the gradient back into image form during
        # backward().
        self._flatten_shape = (h, w, self.n_filters2)
        flatten_dim = h * w * self.n_filters2
        linear = ModularLinearLayer(flatten_dim, self.num_classes, rng=rng, init="xavier")
        return [*block1, *block2, linear]
    # The _forward method performs the forward pass through the entire CNN architecture, passing the
    # input through each layer in sequence and returning the final output logits.
    # Receives: X — a batch of (already standardised) input images, and
    #   training — whether this is a training forward pass (caches what
    #   backward needs) or pure inference.
    # Calculates: runs X through every layer except the last (the
    #   conv/pool/activation stack), flattens the resulting feature map
    #   into a flat vector per sample, then runs that through the final
    #   linear layer.
    # Returns: the raw class logits, shape (batch, num_classes) (not yet
    #   turned into probabilities — that's `_stable_softmax`'s job).
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        out = X
        for layer in self.layers_[:-1]:
            if isinstance(layer, (ConvLayer, PoolingLayer)):
                out = layer(out, training=training)
            else:
                out = layer(out)
        self._flatten_shape = out.shape[1:]
        flat = out.reshape(out.shape[0], -1)
        return self.layers_[-1](flat)
    # The _backward method performs the backward pass through the entire CNN architecture, computing
    # gradients with respect to the input, weights, and biases of each layer in reverse order.
    # Receives: grad_logits — the gradient of the loss with respect to the
    #   network's final output logits.
    # Calculates: backpropagates that gradient through the final linear
    #   layer, reshapes it back from a flat vector into the (H, W, C)
    #   feature-map shape the conv/pool stack expects, then continues
    #   backpropagating through every earlier layer in REVERSE order.
    # Returns: nothing — every layer's own `backward()` call updates that
    #   layer's cached weight/bias gradients as a side effect, ready for
    #   the optimizer step that follows.
    def _backward(self, grad_logits: np.ndarray) -> None:
        linear = self.layers_[-1]
        grad = linear.backward(grad_logits)
        grad = grad.reshape(-1, *self._flatten_shape)
        for layer in reversed(self.layers_[:-1]):
            grad = layer.backward(grad)
    # The _validate_X method checks that the input data X is a 4D array with the expected shape and
    # contains only finite values.
    # Receives: X — raw image data to validate.
    # Calculates: converts X to a float64 array and checks that it's 4D,
    #   matches the expected (H, W, C) shape, and contains no NaN/infinite
    #   values.
    # Returns: the validated float64 array; raises ValueError if any check
    #   fails.
    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 4:
            raise ValueError(f"X must be 4D (n_samples, height, width, channels), got {X.shape}")
        if X.shape[1:] != self.input_shape:
            raise ValueError(f"expected samples of shape {self.input_shape}, got {X.shape[1:]}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")
        return X
    # The fit method trains the CNN on the provided training data X and labels y using mini-batch
    # Adam optimization.
    # Receives: X — a batch of training images, and y — their integer class
    #   labels.
    # Calculates: standardises X (globally, with a single scalar mean/std,
    #   unlike per-feature standardisation), builds the layer stack (unless
    #   one was pre-supplied), then runs mini-batch training for `epochs`
    #   passes: each batch does a forward pass, computes the
    #   softmax+cross-entropy loss and its gradient (which conveniently
    #   simplifies to `probs - target`, skipping an explicit softmax
    #   backward pass), backpropagates through every layer, and updates
    #   every convolutional/linear layer's weights via Adam.
    # Returns: self (the fitted model), storing `layers_`, `loss_history_`,
    #   `n_iter_`, and the standardisation statistics `x_mean_`/`x_scale_`.
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNClassifier":
        """Train the CNN.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, height, width, channels)
        y : np.ndarray of shape (n_samples,)
            Integer class labels in ``[0, num_classes)``.

        Returns
        -------
        self
        """
        X = self._validate_X(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have matching number of samples")
        if y.min() < 0 or y.max() >= self.num_classes:
            msg = f"labels must lie in [0, {self.num_classes}), got range [{y.min()}, {y.max()}]"
            raise ValueError(msg)

        rng = np.random.default_rng(self.random_state)

        # Global-scalar standardisation (one mean/std for the WHOLE image
        # batch, not per-pixel) — appropriate for image data, where every
        # pixel shares roughly the same value range.
        self.x_mean_ = float(X.mean())
        self.x_scale_ = float(X.std())
        if self.x_scale_ < 1e-12:
            self.x_scale_ = 1.0
        X_std = (X - self.x_mean_) / self.x_scale_

        n_samples = X_std.shape[0]
        batch_size = min(self.batch_size, n_samples)
        # One-hot encode every label once, up front, instead of doing it
        # per-batch inside the training loop.
        one_hot = np.eye(self.num_classes)[y]

        self.layers_ = list(self._init_layers) if self._init_layers else self._build_layers(rng)
        param_types = (ConvLayer, ModularLinearLayer)
        # Only these two layer types have learnable weights (ReLU/Pooling/
        # Softmax don't) — these are exactly the layers Adam needs its own
        # moment-tracking arrays for.
        param_layers = [layer for layer in self.layers_ if isinstance(layer, param_types)]

        beta_1, beta_2, epsilon = 0.9, 0.999, 1e-8
        m_w = [np.zeros_like(layer.weight) for layer in param_layers]
        v_w = [np.zeros_like(layer.weight) for layer in param_layers]
        m_b = [np.zeros_like(layer.bias) for layer in param_layers]
        v_b = [np.zeros_like(layer.bias) for layer in param_layers]
        adam_t = 0

        self.loss_history_ = []
        epoch = 0
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(n_samples)
            epoch_loss_sum = 0.0
            for start in range(0, n_samples, batch_size):
                batch_idx = order[start:start + batch_size]
                X_batch = X_std[batch_idx]
                target = one_hot[batch_idx]
                batch_n = X_batch.shape[0]

                logits = self._forward(X_batch)
                probs = _stable_softmax(logits)
                epoch_loss_sum += float(-np.sum(target * np.log(np.clip(probs, 1e-12, 1.0))))

                # softmax + cross-entropy gradient simplifies to probs - target, so there's no need
                # to backprop through a separate softmax layer
                grad_logits = (probs - target) / batch_n
                self._backward(grad_logits)

                adam_t += 1
                bias_correction1 = 1.0 - beta_1 ** adam_t
                bias_correction2 = 1.0 - beta_2 ** adam_t
                for idx, layer in enumerate(param_layers):
                    w_grad = layer.weight_grad
                    b_grad = layer.bias_grad
                    m_w[idx] = beta_1 * m_w[idx] + (1.0 - beta_1) * w_grad
                    v_w[idx] = beta_2 * v_w[idx] + (1.0 - beta_2) * w_grad ** 2
                    m_b[idx] = beta_1 * m_b[idx] + (1.0 - beta_1) * b_grad
                    v_b[idx] = beta_2 * v_b[idx] + (1.0 - beta_2) * b_grad ** 2

                    w_hat = m_w[idx] / bias_correction1
                    v_hat = v_w[idx] / bias_correction2
                    w_step = w_hat / (np.sqrt(v_hat) + epsilon)

                    b_hat = m_b[idx] / bias_correction1
                    vb_hat = v_b[idx] / bias_correction2
                    b_step = b_hat / (np.sqrt(vb_hat) + epsilon)
                    layer.weight -= self.lr * w_step
                    if self.alpha:
                        layer.weight -= self.lr * self.alpha * layer.weight
                    layer.bias -= self.lr * b_step

            self.loss_history_.append(epoch_loss_sum / n_samples)

        self.n_iter_ = epoch
        return self
    # The predict_proba method computes the predicted class probabilities for the input data X by
    # performing a forward pass through the CNN and applying the softmax function to the output
    # logits.
    # Receives: X — new image data to classify.
    # Calculates: standardises X using the TRAINING statistics, runs it
    #   through the fitted network (in pure-inference mode, skipping any
    #   backward-pass caching), then converts the raw logits into
    #   probabilities via softmax.
    # Returns: an array of shape (n_samples, num_classes) — per-class
    #   probabilities, each row summing to 1.
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict per-class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, height, width, channels)

        Returns
        -------
        np.ndarray of shape (n_samples, num_classes)
        """
        if not self.layers_:
            raise RuntimeError("CNNClassifier instance is not fitted yet; call fit() first")
        X = self._validate_X(X)
        X_std = (X - self.x_mean_) / self.x_scale_
        logits = self._forward(X_std, training=False)
        return _stable_softmax(logits)
    # The predict method computes the predicted class labels for the input data X by selecting the
    # class with the highest predicted probability for each sample.
    # Receives: X — new image data to classify.
    # Calculates: the per-class probabilities (via `predict_proba`), then
    #   for each sample, which class has the highest probability.
    # Returns: an array of shape (n_samples,) with the predicted class
    #   label for every sample.
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, height, width, channels)

        Returns
        -------
        np.ndarray of shape (n_samples,) with predicted class labels
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
    # The score method computes the mean classification accuracy on the given test set by comparing
    # the predicted class labels with the true labels and returning the fraction of correctly
    # classified samples.
    # Receives: X — test image data, and y — the true labels for that same
    #   data.
    # Calculates: the model's predictions for X, compared against the true
    #   labels y.
    # Returns: a single float — the fraction of samples classified
    #   correctly.
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean classification accuracy on the given test set.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, height, width, channels)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        float
            Fraction of correctly classified samples in [0, 1].
        """
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))
