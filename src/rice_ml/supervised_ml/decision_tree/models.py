"""
User-friendly decision tree classifier.

This module provides a single estimator:

- ``DecisionTreeClassifier`` for supervised classification

The classifier uses axis-aligned binary splits and supports the ``"gini"``
and ``"entropy"`` criteria.

Examples
--------
>>> X = [[0.0], [1.0], [2.0], [3.0]]
>>> y = [0, 0, 1, 1]
>>> clf = DecisionTreeClassifier(max_depth=1)
>>> _ = clf.fit(X, y)
>>> clf.predict([[0.5], [2.5]]).tolist()
[0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _validate_feature_matrix(
    X: Sequence[Sequence[float]], *, name: str = "X"
) -> np.ndarray:
    """
    Validate a 2D numeric feature matrix.

    Parameters
    ----------
    X : sequence of sequence of float
        Input feature matrix.
    name : str, default="X"
        Variable name used in error messages.

    Returns
    -------
    numpy.ndarray
        Two-dimensional float array.
    """
    try:
        X_arr = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be array-like and numeric.") from exc

    if X_arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array-like structure.")

    if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")

    if not np.all(np.isfinite(X_arr)):
        raise ValueError(f"{name} must contain only finite numeric values.")

    return X_arr


def _validate_target_vector(
    y: Sequence, n_samples: int, *, name: str = "y"
) -> np.ndarray:
    """
    Validate a 1D target vector.

    Parameters
    ----------
    y : sequence
        Target values.
    n_samples : int
        Expected number of samples.
    name : str, default="y"
        Variable name used in error messages.

    Returns
    -------
    numpy.ndarray
        One-dimensional array.
    """
    y_arr = np.asarray(y)

    if y_arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if y_arr.shape[0] != n_samples:
        raise ValueError(f"{name} must have the same number of rows as X.")

    return y_arr


@dataclass
class _TreeNode:
    """A single node in the fitted decision tree."""

    value: object
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        """Return ``True`` when the node has no children."""
        return self.left is None and self.right is None


@dataclass
class DecisionTreeClassifier:
    """
    Decision tree classifier with a small, beginner-friendly API.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        Impurity measure used to choose each split.
    max_depth : int or None, default=None
        Maximum depth of the fitted tree. ``None`` allows the tree to keep
        growing until another stopping rule applies.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required in each leaf.

    Examples
    --------
    >>> X = [[0.0], [1.0], [2.0], [3.0]]
    >>> y = [0, 0, 1, 1]
    >>> model = DecisionTreeClassifier(max_depth=1)
    >>> _ = model.fit(X, y)
    >>> model.predict([[1.1], [2.9]]).tolist()
    [0, 1]
    """

    criterion: str = "gini"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1

    def __post_init__(self) -> None:
        if self.criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be either 'gini' or 'entropy'.")

        if self.max_depth is not None:
            if not isinstance(self.max_depth, int) or self.max_depth <= 0:
                raise ValueError("max_depth must be a positive integer or None.")

        if not isinstance(self.min_samples_split, int) or self.min_samples_split < 2:
            raise ValueError(
                "min_samples_split must be an integer greater than or equal to 2."
            )

        if not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer.")

        self._is_fitted = False

    def fit(
        self, X: Sequence[Sequence[float]], y: Sequence
    ) -> "DecisionTreeClassifier":
        """
        Fit the classifier.

        Parameters
        ----------
        X : sequence of sequence of float
            Training features.
        y : sequence
            Training labels.

        Returns
        -------
        DecisionTreeClassifier
            Fitted classifier.
        """
        self._X = _validate_feature_matrix(X, name="X")
        self._y = _validate_target_vector(y, self._X.shape[0], name="y")

        self.classes_ = np.unique(self._y)
        self.n_features_in_ = self._X.shape[1]
        self.tree_ = self._build_tree(self._X, self._y, depth=0)
        self.tree_depth_ = self._tree_depth(self.tree_)
        self._is_fitted = True
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Predict class labels for a feature matrix.

        Parameters
        ----------
        X : sequence of sequence of float
            Query feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted labels.
        """
        if not self._is_fitted:
            raise ValueError("This model must be fitted before prediction.")

        X_query = _validate_feature_matrix(X, name="X")
        if X_query.shape[1] != self.n_features_in_:
            raise ValueError(
                "X must have the same number of features as the training data."
            )

        predictions = [self._predict_one(row, self.tree_) for row in X_query]
        return np.asarray(predictions)

    def _predict_one(self, x: np.ndarray, node: _TreeNode) -> object:
        """Traverse the tree for a single observation."""
        current = node
        while not current.is_leaf:
            if x[current.feature_index] <= current.threshold:
                current = current.left
            else:
                current = current.right

        return current.value

    def _build_tree(self, X: np.ndarray, y: np.ndarray, *, depth: int) -> _TreeNode:
        """Recursively build the decision tree."""
        node = _TreeNode(value=self._majority_class(y))

        if self._should_stop(y, depth):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature_index, threshold, left_mask = split
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth=depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth=depth + 1)
        return node

    def _should_stop(self, y: np.ndarray, depth: int) -> bool:
        """Decide whether the current node should become a leaf."""
        if np.unique(y).size == 1:
            return True

        if y.size < self.min_samples_split:
            return True

        if self.max_depth is not None and depth >= self.max_depth:
            return True

        return False

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[int, float, np.ndarray] | None:
        """Find the best feature and threshold for a node."""
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)

        if parent_impurity == 0.0:
            return None

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_index in range(n_features):
            values = X[:, feature_index]
            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            sorted_y = y[order]

            split_positions = np.where(np.diff(sorted_values) != 0)[0]
            for split_index in split_positions:
                left_count = split_index + 1
                right_count = n_samples - left_count

                if (
                    left_count < self.min_samples_leaf
                    or right_count < self.min_samples_leaf
                ):
                    continue

                left_y = sorted_y[:left_count]
                right_y = sorted_y[left_count:]
                gain = parent_impurity - self._weighted_child_impurity(
                    left_y, right_y, n_samples
                )
                threshold = float(
                    (sorted_values[split_index] + sorted_values[split_index + 1]) / 2.0
                )

                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        if best_feature is None or best_threshold is None:
            return None

        left_mask = X[:, best_feature] <= best_threshold
        return best_feature, best_threshold, left_mask

    def _weighted_child_impurity(
        self, left_y: np.ndarray, right_y: np.ndarray, n_total: int
    ) -> float:
        """Compute the weighted impurity of a candidate split."""
        left_weight = left_y.size / n_total
        right_weight = right_y.size / n_total
        return left_weight * self._impurity(left_y) + right_weight * self._impurity(
            right_y
        )

    def _impurity(self, y: np.ndarray) -> float:
        """Compute node impurity under the selected criterion."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()

        if self.criterion == "gini":
            return float(1.0 - np.sum(probabilities**2))

        nonzero = probabilities[probabilities > 0]
        return float(-np.sum(nonzero * np.log2(nonzero)))

    def _majority_class(self, y: np.ndarray) -> object:
        """Return the most common class label in ``y``."""
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _tree_depth(self, node: _TreeNode) -> int:
        """Return the depth of the fitted tree."""
        if node.is_leaf:
            return 0

        return 1 + max(self._tree_depth(node.left), self._tree_depth(node.right))
