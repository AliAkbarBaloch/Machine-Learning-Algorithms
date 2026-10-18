"""Fit every mlab model next to its scikit-learn counterpart on the same split.

Run from the repository root:

    python examples/compare_with_sklearn.py

Prints one row per model (test-set score for mlab and scikit-learn, plus
wall-clock fit+predict time) and a Markdown table at the end. All splits use
``random_state=42``; mlab models are seeded where they accept a seed.

The CNN row uses the MNIST CSVs in ``datasets/`` when present (see
datasets/README.md) and falls back to scikit-learn's bundled 8x8 digits.
scikit-learn has no CNN, so that row is compared against ``MLPClassifier``.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from sklearn import cluster as sk_cluster
from sklearn import ensemble as sk_ensemble
from sklearn import linear_model as sk_linear
from sklearn import naive_bayes as sk_nb
from sklearn import neural_network as sk_nn
from sklearn import svm as sk_svm
from sklearn import tree as sk_tree
from sklearn.datasets import load_digits, make_blobs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from mlab.clustering import KMeans, KMeansPlusPlus
from mlab.ensemble import RandomForestClassifier
from mlab.metrics import accuracy_score, adjusted_rand_score, r2_score
from mlab.naive_bayes import GaussianNB, MultinomialNB
from mlab.neural_networks import CNNClassifier, MLPRegressor
from mlab.regression import LinearRegressor, LogisticRegression, SGDClassifier, SGDRegression
from mlab.svm import SVC, SVM, SVR
from mlab.trees import DecisionTreeClassifier

DATA = Path(__file__).resolve().parent.parent / "datasets"
SPLIT = {"test_size": 0.2, "random_state": 42}

Row = Tuple[str, str, str, float, float, float, float]
rows: List[Row] = []


def timed(fn: Callable[[], float]) -> Tuple[float, float]:
    """Return (result of fn, seconds it took)."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def fit_score(model, metric, x_train, x_test, y_train, y_test) -> float:
    """Fit on the training split and score predictions on the test split."""
    model.fit(x_train, y_train)  # not every mlab fit() returns self, so don't chain
    return metric(y_test, model.predict(x_test))


def compare(name, dataset, ours, ref, metric, x_train, x_test, y_train, y_test) -> None:
    """Fit both models on the same split and record their test scores."""
    split = (x_train, x_test, y_train, y_test)
    ours_score, ours_s = timed(lambda: fit_score(ours, metric, *split))
    ref_score, ref_s = timed(lambda: fit_score(ref, metric, *split))
    rows.append((name, dataset, metric.__name__, ours_score, ref_score, ours_s, ref_s))
    print(f"{name:<26} {dataset:<22} mlab={ours_score:.3f}  sklearn={ref_score:.3f}", flush=True)


def standardize(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Scale with training-set statistics only."""
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


def regression() -> None:
    """Crop yield: two features, non-linear fertilizer saturation effect."""
    df = pd.read_csv(DATA / "crop_yield_saturation_data.csv")
    x, y = df[["Rainfall_mm", "Fertilizer_kg_ha"]].to_numpy(), df["Yield_tons_ha"].to_numpy()
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, **SPLIT)
    s_tr, s_te = standardize(x_tr, x_te)
    compare("LinearRegressor", "crop_yield", LinearRegressor(), sk_linear.LinearRegression(),
            r2_score, x_tr, x_te, y_tr, y_te)
    compare("SGDRegression", "crop_yield", SGDRegression(), sk_linear.SGDRegressor(random_state=0),
            r2_score, s_tr, s_te, y_tr, y_te)
    # Closed-form kernel solve is O(n^3): train on the first 2,000 rows.
    compare("SVR (RBF)", "crop_yield (2k train)", SVR(kernel="rbf"), sk_svm.SVR(kernel="rbf"),
            r2_score, s_tr[:2000], s_te, y_tr[:2000], y_te)
    compare("MLPRegressor", "crop_yield", MLPRegressor(random_state=0),
            sk_nn.MLPRegressor(random_state=0, max_iter=100), r2_score, x_tr, x_te, y_tr, y_te)


def linear_classification() -> None:
    """User actions: 65/35 classes, positive class encircled by the negative one."""
    df = pd.read_csv(DATA / "user_actions.csv")
    x, y = df[["metric_A", "metric_B"]].to_numpy(), df["action_taken"].to_numpy()
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, stratify=y, **SPLIT)
    s_tr, s_te = standardize(x_tr, x_te)
    compare("LogisticRegression", "user_actions", LogisticRegression(random_state=0),
            sk_linear.LogisticRegression(), accuracy_score, s_tr, s_te, y_tr, y_te)
    compare("SGDClassifier", "user_actions", SGDClassifier(random_state=0),
            sk_linear.SGDClassifier(loss="log_loss", random_state=0), accuracy_score, s_tr, s_te, y_tr, y_te)
    compare("SVM (linear, primal)", "user_actions", SVM(), sk_svm.LinearSVC(),
            accuracy_score, s_tr, s_te, y_tr, y_te)
    # SMO keeps a full kernel matrix: train on the first 2,000 rows.
    compare("SVC (RBF, SMO)", "user_actions (2k train)", SVC(kernel="rbf", random_state=0),
            sk_svm.SVC(kernel="rbf"), accuracy_score, s_tr[:2000], s_te, y_tr[:2000], y_te)


def naive_bayes_and_kernels() -> None:
    """Sensor XOR (continuous) and support messages (word counts)."""
    df = pd.read_csv(DATA / "sensor_device_types_xor.csv")
    x, y = df[["SensorX", "SensorY"]].to_numpy(), df["DeviceType"].to_numpy()
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, stratify=y, **SPLIT)
    compare("GaussianNB", "sensor_xor", GaussianNB(), sk_nb.GaussianNB(),
            accuracy_score, x_tr, x_te, y_tr, y_te)
    s_tr, s_te = standardize(x_tr, x_te)
    compare("SVC (RBF, SMO)", "sensor_xor", SVC(kernel="rbf", random_state=0), sk_svm.SVC(kernel="rbf"),
            accuracy_score, s_tr, s_te, y_tr, y_te)

    df = pd.read_csv(DATA / "customer_support_messages.csv")
    train, test = train_test_split(df, stratify=df["category"], **SPLIT)
    vectorizer = CountVectorizer()
    x_tr = vectorizer.fit_transform(train["message_text"]).toarray()
    x_te = vectorizer.transform(test["message_text"]).toarray()
    compare("MultinomialNB", "support_messages", MultinomialNB(), sk_nb.MultinomialNB(),
            accuracy_score, x_tr, x_te, train["category"].to_numpy(), test["category"].to_numpy())


def trees() -> None:
    """Customer subscription: mixed numeric/categorical features (one-hot encoded)."""
    df = pd.read_csv(DATA / "customer_subscription_data.csv")
    x = pd.get_dummies(df.drop(columns="Subscribed"), dtype=float).to_numpy()
    y = df["Subscribed"].to_numpy()
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, stratify=y, **SPLIT)
    compare("DecisionTreeClassifier", "subscription", DecisionTreeClassifier(max_depth=5, random_state=0),
            sk_tree.DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=0),
            accuracy_score, x_tr, x_te, y_tr, y_te)
    compare("RandomForestClassifier", "subscription",
            RandomForestClassifier(n_estimators=20, max_depth=5, random_state=0),
            sk_ensemble.RandomForestClassifier(n_estimators=20, max_depth=5, criterion="entropy",
                                               random_state=0),
            accuracy_score, x_tr, x_te, y_tr, y_te)


def clustering() -> None:
    """Four Gaussian blobs with known ground truth, scored by adjusted Rand index."""
    x, y_true = make_blobs(n_samples=1500, centers=4, cluster_std=1.2, random_state=42)
    for name, model, init in [("KMeans", KMeans, "random"), ("KMeansPlusPlus", KMeansPlusPlus, "k-means++")]:
        ours, ours_s = timed(lambda m=model: adjusted_rand_score(y_true, m(n_clusters=4, random_state=0).fit(x).labels_))
        ref, ref_s = timed(lambda i=init: adjusted_rand_score(
            y_true, sk_cluster.KMeans(n_clusters=4, init=i, n_init=1, random_state=0).fit(x).labels_))
        rows.append((name, "make_blobs (4 centers)", "adjusted_rand_score", ours, ref, ours_s, ref_s))
        print(f"{name:<26} {'make_blobs':<22} mlab={ours:.3f}  sklearn={ref:.3f}", flush=True)


def cnn() -> None:
    """Digit images: MNIST CSV subset if available, else sklearn's 8x8 digits."""
    mnist = DATA / "train.csv"
    if mnist.exists():
        df = pd.read_csv(mnist, nrows=6000)
        images = df.drop(columns="label").to_numpy().reshape(-1, 28, 28, 1) / 255.0
        labels, dataset = df["label"].to_numpy(), "MNIST (5k train/1k test)"
        x_tr, x_te, y_tr, y_te = images[:5000], images[5000:], labels[:5000], labels[5000:]
    else:
        digits = load_digits()
        images, labels, dataset = digits.images[..., None] / 16.0, digits.target, "sklearn digits 8x8"
        x_tr, x_te, y_tr, y_te = train_test_split(images, labels, stratify=labels, **SPLIT)
    cnn_model = CNNClassifier(input_shape=x_tr.shape[1:], epochs=5, random_state=0)
    ours, ours_s = timed(lambda: fit_score(cnn_model, accuracy_score, x_tr, x_te, y_tr, y_te))
    mlp = sk_nn.MLPClassifier(random_state=0, max_iter=50)
    flat = (x_tr.reshape(len(x_tr), -1), x_te.reshape(len(x_te), -1), y_tr, y_te)
    ref, ref_s = timed(lambda: fit_score(mlp, accuracy_score, *flat))
    rows.append(("CNNClassifier (5 epochs)", dataset, "accuracy_score", ours, ref, ours_s, ref_s))
    print(f"{'CNNClassifier':<26} {dataset:<22} mlab={ours:.3f}  sklearn MLP={ref:.3f}", flush=True)


def main() -> None:
    """Run every comparison and print a Markdown summary table."""
    warnings.filterwarnings("ignore")  # sklearn convergence warnings at default max_iter
    regression()
    linear_classification()
    naive_bayes_and_kernels()
    trees()
    clustering()
    cnn()
    print("\n| Model | Dataset | Metric | mlab | scikit-learn | mlab time (s) | sklearn time (s) |")
    print("|---|---|---|---:|---:|---:|---:|")
    for name, dataset, metric, ours, ref, ours_s, ref_s in rows:
        print(f"| {name} | {dataset} | {metric} | {ours:.3f} | {ref:.3f} | {ours_s:.2f} | {ref_s:.2f} |")


if __name__ == "__main__":
    main()
