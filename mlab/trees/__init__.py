"""CART-style decision tree classifier and its split/impurity helpers."""

from mlab.trees._decision_tree import DecisionTree, DecisionTreeClassifier
from mlab.trees._tree_utils import entropy, gini_impurity, majority_class

__all__ = ["DecisionTreeClassifier", "DecisionTree", "entropy", "gini_impurity", "majority_class"]
