"""Classification, regression and clustering evaluation metrics."""

from mlab.metrics._classification import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from mlab.metrics._clustering import adjusted_rand_score, davies_bouldin_score, silhouette_score
from mlab.metrics._regression import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    pearson_correlation,
    r2_score,
    regression_report,
    root_mean_squared_error,
    spearman_correlation,
)

__all__ = [
    "accuracy_score",
    "average_precision_score",
    "classification_report",
    "confusion_matrix",
    "f1_score",
    "precision_recall_fscore_support",
    "precision_score",
    "recall_score",
    "roc_auc_score",
    "adjusted_rand_score",
    "davies_bouldin_score",
    "silhouette_score",
    "explained_variance_score",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "pearson_correlation",
    "r2_score",
    "regression_report",
    "root_mean_squared_error",
    "spearman_correlation",
]
