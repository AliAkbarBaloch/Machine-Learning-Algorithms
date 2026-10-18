"""Behavioural tests: each mlab model is checked against scikit-learn or a known answer.

Uses only small synthetic / bundled datasets so the whole suite runs in seconds:

    pytest
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.datasets import load_digits, load_iris, make_blobs, make_moons, make_regression
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.naive_bayes import MultinomialNB as SkMultinomialNB

from mlab import metrics
from mlab.clustering import KMeans, KMeansPlusPlus
from mlab.ensemble import RandomForestClassifier
from mlab.naive_bayes import GaussianNB, MultinomialNB
from mlab.neural_networks import CNNClassifier, MLPRegressor
from mlab.regression import LinearRegressor, LogisticRegression, SGDClassifier, SGDRegression
from mlab.svm import SVC, SVM, SVR
from mlab.trees import DecisionTreeClassifier


@pytest.fixture(name="linear_data")
def fixture_linear_data():
    """Standardised linear-regression problem with a known generating process."""
    x, y = make_regression(n_samples=300, n_features=4, noise=5.0, random_state=0)
    return (x - x.mean(axis=0)) / x.std(axis=0), y


@pytest.fixture(name="binary_data")
def fixture_binary_data():
    """Two well-separated blobs: any working linear classifier should solve it."""
    x, y = make_blobs(n_samples=400, centers=2, cluster_std=1.0, random_state=0)
    return train_test_split((x - x.mean(axis=0)) / x.std(axis=0), y, test_size=0.25, random_state=0)


def test_linear_regressor_matches_sklearn_coefficients(linear_data):
    x, y = linear_data
    model = LinearRegressor()
    model.fit(x, y)
    reference = SkLinearRegression().fit(x, y)
    np.testing.assert_allclose(model.coef_, reference.coef_, rtol=1e-8)
    np.testing.assert_allclose(model.predict(x), reference.predict(x), rtol=1e-8)


@pytest.mark.parametrize("model", [LinearRegressor(), SGDRegression(), LogisticRegression(), SGDClassifier()])
def test_predict_before_fit_raises(model):
    with pytest.raises(ValueError, match="not been fitted"):
        model.predict(np.zeros((2, 2)))


def test_sgd_regression_converges_to_closed_form(linear_data):
    x, y = linear_data
    closed_form = LinearRegressor()
    closed_form.fit(x, y)
    sgd = SGDRegression(learning_rate=0.01, n_iterations=200, batch_size=32)
    sgd.fit(x, y)
    np.testing.assert_allclose(sgd.predict(x), closed_form.predict(x), rtol=0.02, atol=1.0)


@pytest.mark.parametrize("model", [
    LogisticRegression(random_state=0),
    SGDClassifier(n_iterations=100, random_state=0),
    SVM(),
])
def test_linear_classifiers_separate_blobs(model, binary_data):
    x_train, x_test, y_train, y_test = binary_data
    model.fit(x_train, y_train)
    assert metrics.accuracy_score(y_test, model.predict(x_test)) >= 0.9


def test_logistic_probabilities_are_valid(binary_data):
    x_train, x_test, y_train, _ = binary_data
    model = LogisticRegression(random_state=0)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)
    assert proba.shape == (len(x_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("model_cls", [KMeans, KMeansPlusPlus])
def test_kmeans_recovers_blobs(model_cls):
    x, y_true = make_blobs(n_samples=600, centers=3, cluster_std=0.8, random_state=1)
    # A single random initialisation can land in a local optimum (scikit-learn's
    # init="random", n_init=1 does too), so keep the best of 5 runs by inertia,
    # which is what n_init does.
    model = min((model_cls(n_clusters=3, random_state=seed).fit(x) for seed in range(5)),
                key=lambda fitted: fitted.inertia_)
    assert model.cluster_centers_.shape == (3, 2)
    assert metrics.adjusted_rand_score(y_true, model.labels_) > 0.95
    np.testing.assert_array_equal(model.predict(x), model.labels_)


def test_gaussian_nb_matches_sklearn_probabilities():
    x, y = load_iris(return_X_y=True)
    model = GaussianNB()
    model.fit(x, y)
    np.testing.assert_allclose(model.predict_proba(x), SkGaussianNB().fit(x, y).predict_proba(x), atol=1e-6)


def test_multinomial_nb_matches_sklearn_probabilities():
    rng = np.random.default_rng(0)
    x = rng.poisson(2.0, size=(200, 30))
    y = rng.integers(0, 3, size=200)
    model = MultinomialNB(alpha=1.0)
    model.fit(x, y)
    reference = SkMultinomialNB(alpha=1.0).fit(x, y)
    np.testing.assert_allclose(model.predict_proba(x), reference.predict_proba(x), atol=1e-6)


def test_decision_tree_and_forest_on_iris():
    x_train, x_test, y_train, y_test = train_test_split(
        *load_iris(return_X_y=True), test_size=0.3, stratify=load_iris().target, random_state=0)
    tree = DecisionTreeClassifier(max_depth=4, random_state=0)
    tree.fit(x_train, y_train)
    forest = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0)
    forest.fit(x_train, y_train)
    assert metrics.accuracy_score(y_test, tree.predict(x_test)) >= 0.9
    assert metrics.accuracy_score(y_test, forest.predict(x_test)) >= 0.9
    assert tree.get_depth() <= 4
    assert len(forest.trees_) == 20
    np.testing.assert_allclose(forest.feature_importances_.sum(), 1.0)


def test_rbf_svc_learns_nonlinear_boundary():
    x, y = make_moons(n_samples=300, noise=0.15, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    model = SVC(kernel="rbf", C=1.0, random_state=0)
    model.fit(x_train, y_train)
    assert metrics.accuracy_score(y_test, model.predict(x_test)) >= 0.9


def test_svr_fits_sine():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 6, size=(200, 1)), axis=0)
    y = np.sin(x).ravel() + rng.normal(0, 0.1, size=200)
    model = SVR(kernel="rbf", C=10.0)
    model.fit(x, y)
    assert metrics.r2_score(y, model.predict(x)) > 0.9


def test_mlp_regressor_fits_nonlinear_function():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, size=(500, 2))
    y = x[:, 0] ** 2 + np.sin(x[:, 1])
    model = MLPRegressor(hidden_layer_sizes=(32, 16), epochs=200, random_state=0)
    model.fit(x, y)
    assert metrics.r2_score(y, model.predict(x)) > 0.9
    assert len(model.layers_) > 0


def test_cnn_classifies_digits():
    digits = load_digits()
    images = digits.images[..., None] / 16.0
    x_train, x_test, y_train, y_test = train_test_split(
        images, digits.target, test_size=0.25, stratify=digits.target, random_state=0)
    model = CNNClassifier(input_shape=(8, 8, 1), epochs=5, random_state=0)
    model.fit(x_train, y_train)
    assert metrics.accuracy_score(y_test, model.predict(x_test)) >= 0.85


def test_classification_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=300)
    y_pred = np.where(rng.random(300) < 0.7, y_true, rng.integers(0, 3, size=300))
    for average in ["micro", "macro", "weighted"]:
        assert metrics.f1_score(y_true, y_pred, average=average) == pytest.approx(
            skm.f1_score(y_true, y_pred, average=average))
    np.testing.assert_array_equal(metrics.confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred))
    scores = rng.random(300)
    binary = (scores + rng.normal(0, 0.3, 300) > 0.5).astype(int)
    assert metrics.roc_auc_score(binary, scores) == pytest.approx(skm.roc_auc_score(binary, scores))
    assert metrics.average_precision_score(binary, scores) == pytest.approx(
        skm.average_precision_score(binary, scores))


def test_regression_and_clustering_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=200)
    y_pred = y_true + rng.normal(0, 0.5, size=200)
    assert metrics.r2_score(y_true, y_pred) == pytest.approx(skm.r2_score(y_true, y_pred))
    assert metrics.mean_absolute_error(y_true, y_pred) == pytest.approx(skm.mean_absolute_error(y_true, y_pred))
    x, labels = make_blobs(n_samples=200, centers=3, random_state=0)
    assert metrics.silhouette_score(x, labels) == pytest.approx(skm.silhouette_score(x, labels))
    assert metrics.davies_bouldin_score(x, labels) == pytest.approx(skm.davies_bouldin_score(x, labels))
