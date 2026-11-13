"""User-friendly ensemble classifiers.

This module provides two complementary ensemble methods:

- ``RandomForestClassifier`` builds many randomized decision trees in parallel.
- ``AdaBoostClassifier`` builds weighted decision stumps sequentially.

Examples
--------
>>> X = [[0.0], [1.0], [2.0], [3.0]]
>>> y = [0, 0, 1, 1]
>>> forest = RandomForestClassifier(n_estimators=15, max_depth=2, random_state=7)
>>> _ = forest.fit(X, y)
>>> forest.predict([[0.2], [2.8]]).tolist()
[0, 1]
>>> boost = AdaBoostClassifier(n_estimators=10, random_state=7)
>>> _ = boost.fit(X, y)
>>> boost.predict([[0.2], [2.8]]).tolist()
[0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Sequence

import numpy as np

from rice_ml.supervised_ml.decision_tree.models import (
    _validate_feature_matrix,
    _validate_target_vector,
)


@dataclass
class _ForestNode:
    """One node in a randomized classification tree."""

    probabilities: np.ndarray
    feature_index: int | None = None
    threshold: float | None = None
    left: "_ForestNode | None" = None
    right: "_ForestNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class _RandomizedTree:
    """Small internal classification tree used by the random forest."""

    def __init__(
        self,
        *,
        n_classes: int,
        max_depth: int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: int,
        rng: np.random.Generator,
    ) -> None:
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RandomizedTree":
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([self._predict_one(row) for row in X])

    def _predict_one(self, row: np.ndarray) -> np.ndarray:
        node = self.root_
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.probabilities

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _ForestNode:
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        node = _ForestNode(probabilities=counts / counts.sum())

        if (
            np.count_nonzero(counts) == 1
            or y.size < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature, threshold, mask, gain = split
        self.feature_importances_[feature] += y.size * gain
        node.feature_index = feature
        node.threshold = threshold
        node.left = self._build_tree(X[mask], y[mask], depth + 1)
        node.right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return node

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[int, float, np.ndarray, float] | None:
        parent_impurity = self._gini(y)
        features = self.rng.choice(X.shape[1], size=self.max_features, replace=False)
        best: tuple[int, float, np.ndarray, float] | None = None
        best_gain = 0.0

        for feature in features:
            values = X[:, feature]
            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            sorted_y = y[order]

            for split_index in np.where(np.diff(sorted_values) != 0)[0]:
                left_count = split_index + 1
                right_count = y.size - left_count
                if (
                    left_count < self.min_samples_leaf
                    or right_count < self.min_samples_leaf
                ):
                    continue

                left_y = sorted_y[:left_count]
                right_y = sorted_y[left_count:]
                child_impurity = (
                    left_count * self._gini(left_y) + right_count * self._gini(right_y)
                ) / y.size
                gain = parent_impurity - child_impurity

                if gain > best_gain + 1e-12:
                    threshold = float(
                        (sorted_values[split_index] + sorted_values[split_index + 1])
                        / 2
                    )
                    mask = values <= threshold
                    best = int(feature), threshold, mask, float(gain)
                    best_gain = gain

        return best

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        counts = np.bincount(y)
        probabilities = counts[counts > 0] / y.size
        return float(1.0 - np.sum(probabilities**2))


def _resolve_max_features(
    max_features: str | int | float | None, n_features: int
) -> int:
    """Convert a user-facing feature rule into a feature count."""
    if max_features is None:
        return n_features
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if max_features == "log2":
        return max(1, int(np.log2(n_features)))
    if isinstance(max_features, int) and not isinstance(max_features, bool):
        if 1 <= max_features <= n_features:
            return max_features
        raise ValueError(
            "integer max_features must be between 1 and the number of features."
        )
    if isinstance(max_features, float):
        if 0.0 < max_features <= 1.0:
            return max(1, int(np.ceil(max_features * n_features)))
        raise ValueError("float max_features must be in the interval (0, 1].")
    raise ValueError("max_features must be 'sqrt', 'log2', an int, a float, or None.")


@dataclass
class RandomForestClassifier:
    """Random forest classifier using bootstrap samples and randomized splits.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required in each leaf.
    max_features : {"sqrt", "log2"}, int, float, or None, default="sqrt"
        Features considered at each split. A float represents a fraction.
    bootstrap : bool, default=True
        Whether each tree trains on a sample drawn with replacement.
    random_state : int or None, default=None
        Seed used to make the forest reproducible.
    """

    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | int | float | None = "sqrt"
    bootstrap: bool = True
    random_state: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.n_estimators, int)
            or isinstance(self.n_estimators, bool)
            or self.n_estimators <= 0
        ):
            raise ValueError("n_estimators must be a positive integer.")
        if self.max_depth is not None and (
            not isinstance(self.max_depth, int)
            or isinstance(self.max_depth, bool)
            or self.max_depth <= 0
        ):
            raise ValueError("max_depth must be a positive integer or None.")
        if not isinstance(self.min_samples_split, int) or self.min_samples_split < 2:
            raise ValueError(
                "min_samples_split must be an integer greater than or equal to 2."
            )
        if not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer.")
        if not isinstance(self.bootstrap, bool):
            raise TypeError("bootstrap must be a boolean.")
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise TypeError("random_state must be an integer or None.")
        self._is_fitted = False

    def fit(
        self, X: Sequence[Sequence[float]], y: Sequence
    ) -> "RandomForestClassifier":
        """Fit the random forest and return the fitted classifier."""
        X_arr = _validate_feature_matrix(X)
        y_arr = _validate_target_vector(y, X_arr.shape[0])
        self.classes_, y_encoded = np.unique(y_arr, return_inverse=True)
        self.n_features_in_ = X_arr.shape[1]
        self.max_features_ = _resolve_max_features(
            self.max_features, self.n_features_in_
        )

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        importances = np.zeros(self.n_features_in_, dtype=float)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                sample_indices = rng.integers(0, X_arr.shape[0], size=X_arr.shape[0])
            else:
                sample_indices = np.arange(X_arr.shape[0])

            tree = _RandomizedTree(
                n_classes=self.classes_.size,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features_,
                rng=rng,
            ).fit(X_arr[sample_indices], y_encoded[sample_indices])
            self.estimators_.append(tree)
            importances += tree.feature_importances_

        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        self._is_fitted = True
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Predict class probabilities by averaging probabilities across trees."""
        X_arr = self._validate_prediction_data(X)
        probabilities = np.zeros((X_arr.shape[0], self.classes_.size), dtype=float)
        for tree in self.estimators_:
            probabilities += tree.predict_proba(X_arr)
        return probabilities / len(self.estimators_)

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Predict class labels by majority probability across trees."""
        indices = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[indices]

    def _validate_prediction_data(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("This model must be fitted before prediction.")
        X_arr = _validate_feature_matrix(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                "X must have the same number of features as the training data."
            )
        return X_arr


@dataclass
class _DecisionStump:
    """A one-level weighted decision tree used by AdaBoost."""

    feature_index: int | None
    threshold: float | None
    left_class: int
    right_class: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.feature_index is None:
            return np.full(X.shape[0], self.left_class, dtype=int)
        return np.where(
            X[:, self.feature_index] <= self.threshold,
            self.left_class,
            self.right_class,
        )


def _weighted_majority(y: np.ndarray, weights: np.ndarray, n_classes: int) -> int:
    scores = np.bincount(y, weights=weights, minlength=n_classes)
    return int(np.argmax(scores))


def _fit_stump(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, n_classes: int
) -> tuple[_DecisionStump, np.ndarray, float]:
    """Fit the decision stump with the smallest weighted error."""
    majority = _weighted_majority(y, weights, n_classes)
    best_stump = _DecisionStump(None, None, majority, majority)
    best_predictions = best_stump.predict(X)
    best_error = float(weights[best_predictions != y].sum())

    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        if values.size < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2

        for threshold in thresholds:
            left = X[:, feature] <= threshold
            right = ~left
            left_class = _weighted_majority(y[left], weights[left], n_classes)
            right_class = _weighted_majority(y[right], weights[right], n_classes)
            predictions = np.where(left, left_class, right_class)
            error = float(weights[predictions != y].sum())

            if error < best_error - 1e-12:
                best_error = error
                best_predictions = predictions
                best_stump = _DecisionStump(
                    feature_index=feature,
                    threshold=float(threshold),
                    left_class=left_class,
                    right_class=right_class,
                )

    return best_stump, best_predictions, best_error


@dataclass
class AdaBoostClassifier:
    """AdaBoost classifier using weighted decision stumps.

    This implementation uses the multiclass SAMME algorithm, so it supports
    both binary and multiclass targets.

    Parameters
    ----------
    n_estimators : int, default=50
        Maximum number of decision stumps to train.
    learning_rate : float, default=1.0
        Contribution made by each fitted stump.
    random_state : int or None, default=None
        Accepted for a familiar, reproducible estimator API. Stump fitting is
        deterministic, so the value does not change results in this version.
    """

    n_estimators: int = 50
    learning_rate: float = 1.0
    random_state: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.n_estimators, int)
            or isinstance(self.n_estimators, bool)
            or self.n_estimators <= 0
        ):
            raise ValueError("n_estimators must be a positive integer.")
        if (
            not isinstance(self.learning_rate, (int, float))
            or isinstance(self.learning_rate, bool)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a positive number.")
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise TypeError("random_state must be an integer or None.")
        self._is_fitted = False

    def fit(self, X: Sequence[Sequence[float]], y: Sequence) -> "AdaBoostClassifier":
        """Fit weighted decision stumps sequentially and return the model."""
        X_arr = _validate_feature_matrix(X)
        y_arr = _validate_target_vector(y, X_arr.shape[0])
        self.classes_, y_encoded = np.unique(y_arr, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("y must contain at least two classes.")

        self.n_features_in_ = X_arr.shape[1]
        n_samples = X_arr.shape[0]
        n_classes = self.classes_.size
        weights = np.full(n_samples, 1.0 / n_samples)
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        importances = np.zeros(self.n_features_in_, dtype=float)

        for _ in range(self.n_estimators):
            stump, predictions, error = _fit_stump(X_arr, y_encoded, weights, n_classes)
            if error >= 1.0 - (1.0 / n_classes):
                break

            clipped_error = float(np.clip(error, 1e-12, 1.0 - 1e-12))
            alpha = self.learning_rate * (
                log((1.0 - clipped_error) / clipped_error) + log(n_classes - 1)
            )
            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(error)
            if stump.feature_index is not None:
                importances[stump.feature_index] += alpha

            incorrect = predictions != y_encoded
            weights *= np.exp(alpha * incorrect)
            weights /= weights.sum()
            if error <= 1e-12:
                break

        self.estimator_weights_ = np.asarray(self.estimator_weights_, dtype=float)
        self.estimator_errors_ = np.asarray(self.estimator_errors_, dtype=float)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        self._fallback_class = _weighted_majority(
            y_encoded, np.ones(n_samples), n_classes
        )
        self._is_fitted = True
        return self

    def decision_function(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Return one weighted ensemble score per class."""
        X_arr = self._validate_prediction_data(X)
        scores = np.zeros((X_arr.shape[0], self.classes_.size), dtype=float)
        if not self.estimators_:
            scores[:, self._fallback_class] = 1.0
            return scores

        rows = np.arange(X_arr.shape[0])
        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            scores[rows, stump.predict(X_arr)] += alpha
        return scores

    def predict_proba(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Convert ensemble scores into normalized class probabilities."""
        scores = self.decision_function(X)
        scores = scores / max(self.classes_.size - 1, 1)
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Predict the class receiving the largest weighted vote."""
        indices = np.argmax(self.decision_function(X), axis=1)
        return self.classes_[indices]

    def _validate_prediction_data(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("This model must be fitted before prediction.")
        X_arr = _validate_feature_matrix(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                "X must have the same number of features as the training data."
            )
        return X_arr
