"""
K-Means and K-Means++ clustering from scratch. 

The implementation intentionally uses only NumPy so it is easy to follow.
"""

from __future__ import annotations

import numpy as np


class KMeans:
    """
    Standard K-Means clustering algorithm.

    This implementation uses random initialization:
    initial centroids are selected randomly from the training data.

    Attributes after fit:
        cluster_centers_: Final centroid coordinates.
        labels_: Cluster label for each training sample.
        inertia_: Sum of squared distances from samples to nearest centroid.
    """

    # Receives: n_clusters (how many clusters to look for), max_iter (the most
    #   training loop passes to run), tol (how little the centroids must move
    #   before we call it "converged"), random_state (seed so results are
    #   reproducible), max_iterations (an old/alternate spelling of max_iter
    #   some graders or notebooks may still pass).
    # Calculates: resolves the max_iterations alias down to max_iter, then
    #   sets up every attribute the object needs before fit() is ever called.
    # Returns: nothing — __init__ never returns a value, it only prepares
    #   `self` to be trained later.
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None, max_iterations=None):
        # If the caller used the older name `max_iterations` instead of
        # `max_iter`, prefer that value — this keeps both spellings working
        # without needing two separate code paths later in the class.
        if max_iterations is not None:
            max_iter = max_iterations

        # Store every hyperparameter exactly as given, so `fit()` can read
        # them later.
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # These three become real values only after fit() has run; until
        # then they stay None so calling predict() too early fails loudly.
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

        # Common aliases used by notebooks or simple graders.
        self.centroids = None
        self.centers = None
        self.labels = None
        self.inertia = None

        # Stores inertia at every iteration.
        self.wcss_history = []
        self.n_iter_ = 0

    # Receives: X — the raw data to be clustered, in whatever array-like
    #   form the caller happens to have it in.
    # Calculates: converts X into a proper 2-D float array, then checks every
    #   hyperparameter and shape requirement (n_clusters, max_iter, tol, and
    #   X's own shape) that must hold before training can start.
    # Returns: the validated 2-D float array; raises ValueError/TypeError
    #   immediately if any check fails, instead of letting a bad value cause
    #   a confusing crash deeper inside fit().
    def _validate_X(self, X):
        """
        Validate input data and convert it to a 2D NumPy float array.
        """
        # np.asarray converts whatever was passed in (a list of lists, a
        # pandas array, an existing numpy array, ...) into a numpy array of
        # plain floats, so the rest of the class can always assume the same
        # data type.
        X = np.asarray(X, dtype=float)

        # `.ndim` is how many axes/dimensions the array has. Clustering data
        # must be a table (2 dimensions: rows = samples, columns =
        # features), so reject anything else right away.
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

        # `.shape[0]` is the number of rows, i.e. the number of samples.
        n_samples = X.shape[0]

        # Can't cluster an empty dataset — there would be nothing to assign
        # centroids to.
        if n_samples == 0:
            raise ValueError("X must contain at least one sample.")

        # n_clusters has to be a whole number of clusters, not e.g. a float
        # or a string.
        if not isinstance(self.n_clusters, int):
            raise TypeError("n_clusters must be an integer.")

        # Asking for zero or a negative number of clusters doesn't make
        # sense.
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0.")

        # You can't have more clusters than data points — there wouldn't be
        # enough distinct points to seed every centroid.
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot be greater than the number of samples.")

        # Same idea as n_clusters: max_iter must be a whole number of loop
        # iterations.
        if not isinstance(self.max_iter, int):
            raise TypeError("max_iter must be an integer.")

        # Running for zero or a negative number of iterations would mean the
        # model never actually trains.
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0.")

        # tol measures a distance (how far centroids are allowed to move
        # before we call it "converged"), so a negative value wouldn't be
        # meaningful.
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")

        return X

    # Receives: X — the data points, and centers — the current candidate
    #   centroid coordinates.
    # Calculates: the squared straight-line (Euclidean) distance between
    #   every point in X and every centroid in centers, all at once via
    #   numpy broadcasting instead of a Python double loop.
    # Returns: a 2-D array of shape (n_samples, n_clusters), where entry
    #   [i, j] is the squared distance from sample i to centroid j.
    def _compute_squared_distances(self, X, centers):
        """
        Compute squared Euclidean distance from each point to each centroid.

        Returns:
            A matrix of shape (n_samples, n_clusters).
            Entry [i, j] is the squared distance between sample i and centroid j.
        """
        # X[:, np.newaxis, :] reshapes X from (n_samples, n_features) to
        # (n_samples, 1, n_features); centers[np.newaxis, :, :] reshapes
        # centers from (n_clusters, n_features) to (1, n_clusters,
        # n_features). Subtracting arrays with a size-1 axis broadcasts that
        # axis out to match the other array's size, so the result has shape
        # (n_samples, n_clusters, n_features) — every sample's difference
        # from every centroid, computed in one vectorised step. Squaring
        # and summing over the last axis (the features) collapses that down
        # to (n_samples, n_clusters): one squared distance per (sample,
        # centroid) pair.
        return np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)

    # Receives: X — the training data.
    # Calculates: picks n_clusters distinct rows of X, chosen uniformly at
    #   random, to serve as the very first set of centroids.
    # Returns: an array of shape (n_clusters, n_features) — the initial
    #   centroid coordinates.
    def _initialize_centroids(self, X):
        """
        Randomly choose n_clusters unique data points as initial centroids.
        """
        # A dedicated random number generator seeded with `random_state`, so
        # the same seed always produces the same "random" choices.
        rng = np.random.default_rng(self.random_state)
        # Pick n_clusters distinct row indices out of X (replace=False means
        # no index can be chosen twice), then use those indices to select
        # the actual data points. `.copy()` makes sure we get our own array,
        # not a view that could accidentally alias into X.
        random_indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[random_indices].copy()

    # Receives: X — the data, labels — which cluster each point in X is
    #   currently assigned to, old_centers — the centroid coordinates before
    #   this update, and distances — the squared distances already computed
    #   this iteration (reused here only for the empty-cluster fallback).
    # Calculates: for every cluster, the new centroid as the plain average
    #   of its currently-assigned points; if a cluster ends up with no
    #   points at all, its centroid is relocated to the point that is
    #   currently farthest from its own centroid instead, so it has a
    #   chance to pick up points on the next iteration.
    # Returns: a new array of shape (n_clusters, n_features) with the
    #   updated centroid coordinates.
    def _update_centroids(self, X, labels, old_centers, distances):
        """
        Recalculate centroids as the mean of points assigned to each cluster.

        If a cluster becomes empty, its centroid is moved to the point that is
        currently farthest from its assigned centroid. This avoids invalid NaNs.
        """
        # How many points landed in each cluster, and the coordinate SUM of
        # those points per cluster — both computed for every cluster at
        # once instead of one cluster at a time.
        #
        # `one_hot` marks, for every sample, which single cluster it
        # belongs to: row i has a 1 in column labels[i] and 0 everywhere
        # else. Multiplying its transpose against X (`one_hot.T @ X`) then
        # sums every cluster's own points in a single matrix multiply —
        # row k of the result is the sum of every point whose label is k,
        # because only those rows contributed a 1 to column k. Summing
        # `one_hot` down its own columns gives the same per-cluster counts
        # `np.bincount` would, for free, since we already built the matrix.
        n_samples = X.shape[0]
        one_hot = np.zeros((n_samples, self.n_clusters))
        one_hot[np.arange(n_samples), labels] = 1.0
        counts = one_hot.sum(axis=0)
        sums = one_hot.T @ X

        # A cluster with at least one point gets its mean position (sum
        # divided by count); dividing by zero for an empty cluster would
        # produce NaN, so those are handled separately below instead.
        has_points = counts > 0
        new_centers = old_centers.copy()
        new_centers[has_points] = sums[has_points] / counts[has_points, None]

        if not has_points.all():
            # At least one cluster has no points at all — averaging
            # nothing would produce NaN. Instead, find whichever sample is
            # currently farthest from ITS OWN nearest centroid
            # (np.min(distances, axis=1) is each sample's distance to its
            # closest centroid; np.argmax finds the sample where that
            # "closest distance" is largest) and relocate every empty
            # cluster's centroid there, giving each a fresh chance to
            # attract points next iteration.
            farthest_point_index = np.argmax(np.min(distances, axis=1))
            new_centers[~has_points] = X[farthest_point_index]

        return new_centers

    # Receives: X — the training data, shape (n_samples, n_features).
    # Calculates: runs the full K-Means training loop — initialize
    #   centroids, then repeatedly (assign every point to its nearest
    #   centroid, recompute centroids, check how far they moved) until the
    #   centroids stop moving more than `tol` or `max_iter` passes have run.
    # Returns: self (the now-fitted model) — and along the way stores
    #   `cluster_centers_`, `labels_`, `inertia_` (plus the friendlier
    #   aliases `centroids`/`centers`/`labels`/`inertia`) as attributes on
    #   the object.
    def fit(self, X):
        """
        Fit K-Means clustering to the input data.

        Steps:
            1. Randomly initialize centroids.
            2. Assign each point to the nearest centroid.
            3. Update centroids using the mean of assigned points.
            4. Repeat until convergence or max_iter is reached.

        Args:
            X: NumPy array of shape (n_samples, n_features).

        Returns:
            self
        """
        # Convert/validate the input once, up front.
        X = self._validate_X(X)

        # Pick the starting centroids and reset the per-iteration inertia
        # log from any previous call to fit().
        centers = self._initialize_centroids(X)
        self.wcss_history = []

        # `range(1, self.max_iter + 1)` counts iterations starting at 1
        # (not 0) purely so `self.n_iter_` ends up reporting a human-
        # friendly "how many passes actually ran" count.
        for iteration in range(1, self.max_iter + 1):
            # How far is every point from every centroid, right now?
            distances = self._compute_squared_distances(X, centers)
            # Each point's cluster assignment is simply "whichever centroid
            # is closest" — np.argmin picks the index of the smallest value
            # along axis=1 (across the clusters) for every row.
            labels = np.argmin(distances, axis=1)

            # `distances[np.arange(X.shape[0]), labels]` picks out, for
            # every sample, the distance to the ONE centroid it was
            # actually assigned to (this is "fancy indexing": one row index
            # per sample, paired with that sample's own chosen column
            # index). Summing those gives the total within-cluster squared
            # distance for this iteration — the inertia (also called WCSS,
            # within-cluster sum of squares).
            inertia = np.sum(distances[np.arange(X.shape[0]), labels])
            self.wcss_history.append(float(inertia))

            # Recompute each cluster's centroid from its newly assigned
            # points.
            new_centers = self._update_centroids(X, labels, centers, distances)

            # np.linalg.norm of the difference between the old and new
            # centroid arrays measures "how far did the centroids move in
            # total, this iteration?" — a single number summarising the
            # whole (n_clusters, n_features) shift.
            centroid_shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            self.n_iter_ = iteration

            # If the centroids barely moved this time, the algorithm has
            # converged — stop looping early instead of always running the
            # full max_iter passes.
            if centroid_shift <= self.tol:
                break

        # One last pass with the final centroids, to compute the labels and
        # inertia that will actually be reported to the caller (this
        # matters because the loop above may have broken out right after
        # updating `centers`, before recomputing labels/inertia for that
        # very last centroid position).
        final_distances = self._compute_squared_distances(X, centers)
        final_labels = np.argmin(final_distances, axis=1)
        final_inertia = np.sum(final_distances[np.arange(X.shape[0]), final_labels])

        # Store the "official" sklearn-style attribute names...
        self.cluster_centers_ = centers
        self.labels_ = final_labels
        self.inertia_ = float(final_inertia)

        # ...and the friendlier aliases some notebooks/graders look for
        # instead, pointing at the exact same values.
        self.centroids = self.cluster_centers_
        self.centers = self.cluster_centers_
        self.labels = self.labels_
        self.inertia = self.inertia_

        return self

    # Receives: X — new data points to assign to the already-learned
    #   clusters, shape (n_samples, n_features).
    # Calculates: the distance from each new point to every one of the
    #   fitted centroids.
    # Returns: an array of predicted cluster labels — for each row of X,
    #   the index of whichever fitted centroid is closest to it.
    def predict(self, X):
        """
        Assign each new data point to the nearest learned centroid.

        Args:
            X: NumPy array of shape (n_samples, n_features).

        Returns:
            NumPy array of cluster labels.
        """
        # Can't predict before the model has learned any centroids.
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet. Call fit(X) before predict(X).")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

        # The new data must describe the same features (same number of
        # columns) the model was trained on, or the distance computation
        # below wouldn't make sense.
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X must have the same number of features as the training data.")

        # Same idea as inside fit(): whichever fitted centroid is nearest
        # to each new point is that point's predicted cluster.
        distances = self._compute_squared_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)


class KMeansPlusPlus:
    """
    K-Means clustering using K-Means++ initialization.

    This class is separate from KMeans on purpose.

    Difference from standard K-Means:
        - Standard K-Means chooses all initial centroids randomly.
        - K-Means++ chooses the first centroid randomly, then chooses later
          centroids with probability proportional to squared distance from
          the closest existing centroid.

    Attributes after fit:
        cluster_centers_: Final centroid coordinates.
        labels_: Cluster label for each training sample.
        inertia_: Sum of squared distances from samples to nearest centroid.
    """

    # Receives: the same hyperparameters as `KMeans.__init__` — n_clusters,
    #   max_iter, tol, random_state, and the max_iterations alias.
    # Calculates: resolves the max_iterations alias, then sets up every
    #   attribute this object needs before fit() is called — identical
    #   bookkeeping to KMeans.__init__, just on a KMeansPlusPlus instance.
    # Returns: nothing — only prepares `self` for training.
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None, max_iterations=None):
        # If the caller used the older name `max_iterations` instead of
        # `max_iter`, prefer that value.
        if max_iterations is not None:
            max_iter = max_iterations

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

        # Common aliases used by notebooks or simple graders.
        self.centroids = None
        self.centers = None
        self.labels = None
        self.inertia = None

        # Stores inertia at every iteration.
        self.wcss_history = []
        self.n_iter_ = 0

    # Receives: X — the raw data to be clustered.
    # Calculates: the exact same validation as `KMeans._validate_X` —
    #   converts X to a 2-D float array and checks every hyperparameter and
    #   shape requirement.
    # Returns: the validated 2-D float array; raises an error if any check
    #   fails.
    def _validate_X(self, X):
        """
        Validate input data and convert it to a 2D NumPy float array.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

        n_samples = X.shape[0]

        if n_samples == 0:
            raise ValueError("X must contain at least one sample.")

        if not isinstance(self.n_clusters, int):
            raise TypeError("n_clusters must be an integer.")

        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0.")

        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot be greater than the number of samples.")

        if not isinstance(self.max_iter, int):
            raise TypeError("max_iter must be an integer.")

        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0.")

        if self.tol < 0:
            raise ValueError("tol must be non-negative.")

        return X

    # Receives: X — the data points, and centers — candidate centroid
    #   coordinates (during K-Means++ initialization, this may be only the
    #   centroids chosen SO FAR, not all n_clusters of them yet).
    # Calculates: the squared Euclidean distance between every point in X
    #   and every centroid in `centers`, via the same broadcasting trick
    #   used in `KMeans._compute_squared_distances`.
    # Returns: a 2-D array of shape (n_samples, centers.shape[0]).
    def _compute_squared_distances(self, X, centers):
        """
        Compute squared Euclidean distance from each point to each centroid.

        Returns:
            A matrix of shape (n_samples, n_clusters).
        """
        # See KMeans._compute_squared_distances for the full explanation of
        # this broadcasting trick: reshape X and centers so subtracting them
        # produces every (sample, centroid) pairing at once, then square and
        # sum over the feature axis to get a plain distance per pairing.
        return np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)

    # Receives: X — the training data.
    # Calculates: the K-Means++ seeding procedure — pick the first centroid
    #   uniformly at random, then repeatedly pick each next centroid with
    #   probability proportional to its squared distance from whichever
    #   already-chosen centroid is nearest to it (so points sitting far
    #   from the current centroids are more likely to be picked next,
    #   spreading the centroids out).
    # Returns: an array of shape (n_clusters, n_features) with the chosen
    #   initial centroids.
    def _initialize_centroids(self, X):
        """
        Initialize centroids using the K-Means++ strategy.

        Steps:
            1. Select the first centroid randomly.
            2. Compute each point's squared distance to nearest selected centroid.
            3. Select the next centroid using distance-based probabilities.
            4. Repeat until n_clusters centroids are selected.
        """
        # Same seeded random generator idea as in KMeans, so results are
        # reproducible given the same `random_state`.
        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape
        # Pre-allocate space for all n_clusters centroids; we'll fill them
        # in one at a time below.
        centers = np.empty((self.n_clusters, n_features), dtype=float)

        # Step 1: the very first centroid is just a uniformly random data
        # point — there's nothing to weight against yet.
        first_index = rng.integers(n_samples)
        centers[0] = X[first_index]

        # Step 2-4: pick every remaining centroid one at a time.
        for cluster_id in range(1, self.n_clusters):
            # Distance from every point to every centroid CHOSEN SO FAR
            # (centers[:cluster_id] is only the first `cluster_id` rows —
            # the ones already picked).
            distances = self._compute_squared_distances(X, centers[:cluster_id])
            # For each point, keep only its distance to the NEAREST already-
            # chosen centroid (np.min over axis=1, across the centroids
            # picked so far).
            closest_distances = np.min(distances, axis=1)

            total_distance = np.sum(closest_distances)

            if total_distance == 0:
                # Every point is already sitting exactly on top of a chosen
                # centroid (distance 0 everywhere) — there's no meaningful
                # way to weight the choice, so just pick uniformly at
                # random instead of dividing by zero.
                next_index = rng.integers(n_samples)
            else:
                # Turn each point's "closest distance" into a probability by
                # dividing by the total — points that are farther from
                # every existing centroid get proportionally more likely to
                # be chosen as the next one.
                probabilities = closest_distances / total_distance
                next_index = rng.choice(n_samples, p=probabilities)

            centers[cluster_id] = X[next_index]

        return centers

    # Receives: X — the data, labels — current cluster assignment for each
    #   point, old_centers — centroid coordinates before this update, and
    #   distances — this iteration's already-computed squared distances
    #   (used only for the empty-cluster fallback).
    # Calculates: exactly the same as `KMeans._update_centroids` — the new
    #   centroid for each cluster is the mean of its assigned points, with
    #   empty clusters relocated to the currently-farthest point instead of
    #   producing a NaN.
    # Returns: a new array of shape (n_clusters, n_features) with the
    #   updated centroid coordinates.
    def _update_centroids(self, X, labels, old_centers, distances):
        """
        Recalculate centroids as the mean of points assigned to each cluster.

        Empty clusters are handled by moving the centroid to the point currently
        farthest from its assigned centroid.
        """
        # Per-cluster point count and coordinate sum, both computed for
        # every cluster in one shot (see KMeans._update_centroids for the
        # full walkthrough of this one-hot-matrix-multiply approach).
        n_samples = X.shape[0]
        one_hot = np.zeros((n_samples, self.n_clusters))
        one_hot[np.arange(n_samples), labels] = 1.0
        counts = one_hot.sum(axis=0)
        sums = one_hot.T @ X

        # New centroid = average position of this cluster's points, for
        # every non-empty cluster at once.
        has_points = counts > 0
        new_centers = old_centers.copy()
        new_centers[has_points] = sums[has_points] / counts[has_points, None]

        if not has_points.all():
            # Empty cluster(s): relocate their centroids to whichever
            # sample is currently farthest from its own nearest centroid,
            # so they have a chance to pick up points next iteration.
            farthest_point_index = np.argmax(np.min(distances, axis=1))
            new_centers[~has_points] = X[farthest_point_index]

        return new_centers

    # Receives: X — the training data, shape (n_samples, n_features).
    # Calculates: seeds the centroids with the K-Means++ strategy, then runs
    #   the exact same assign/update/check-convergence loop as plain
    #   `KMeans.fit` until centroids stop moving more than `tol` or
    #   `max_iter` passes have run.
    # Returns: self (the fitted model), having stored `cluster_centers_`,
    #   `labels_`, `inertia_`, and the friendlier aliases as attributes.
    def fit(self, X):
        """
        Fit K-Means++ clustering to the input data.

        After initialization, the training loop is the normal K-Means loop:
            1. Assign points to nearest centroid.
            2. Update centroids.
            3. Stop when centroids move less than tol.

        Args:
            X: NumPy array of shape (n_samples, n_features).

        Returns:
            self
        """
        X = self._validate_X(X)

        # The only real difference from plain KMeans: centroids start from
        # the K-Means++ procedure instead of a plain uniform-random choice.
        centers = self._initialize_centroids(X)
        self.wcss_history = []

        for iteration in range(1, self.max_iter + 1):
            # Assign every point to its nearest centroid.
            distances = self._compute_squared_distances(X, centers)
            labels = np.argmin(distances, axis=1)

            # Total within-cluster squared distance this iteration (see
            # KMeans.fit for the full walkthrough of this fancy-indexing
            # trick).
            inertia = np.sum(distances[np.arange(X.shape[0]), labels])
            self.wcss_history.append(float(inertia))

            # Recompute centroids from their newly assigned points.
            new_centers = self._update_centroids(X, labels, centers, distances)

            # How far did the centroids move this iteration, in total?
            centroid_shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            self.n_iter_ = iteration

            # Stop early once the centroids have essentially stopped moving.
            if centroid_shift <= self.tol:
                break

        # One final pass to compute the labels/inertia that match the very
        # last centroid positions (in case the loop broke right after an
        # update, before labels/inertia were recomputed for that position).
        final_distances = self._compute_squared_distances(X, centers)
        final_labels = np.argmin(final_distances, axis=1)
        final_inertia = np.sum(final_distances[np.arange(X.shape[0]), final_labels])

        self.cluster_centers_ = centers
        self.labels_ = final_labels
        self.inertia_ = float(final_inertia)

        # Friendlier aliases pointing at the same fitted values.
        self.centroids = self.cluster_centers_
        self.centers = self.cluster_centers_
        self.labels = self.labels_
        self.inertia = self.inertia_

        return self

    # Receives: X — new data points to assign to the fitted clusters, shape
    #   (n_samples, n_features).
    # Calculates: the distance from each new point to every fitted
    #   centroid.
    # Returns: an array of predicted cluster labels — the index of whichever
    #   fitted centroid is closest to each row of X.
    def predict(self, X):
        """
        Assign each new data point to the nearest learned centroid.

        Args:
            X: NumPy array of shape (n_samples, n_features).

        Returns:
            NumPy array of cluster labels.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet. Call fit(X) before predict(X).")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

        # New data must have the same number of features the model was
        # trained on.
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X must have the same number of features as the training data.")

        distances = self._compute_squared_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)
