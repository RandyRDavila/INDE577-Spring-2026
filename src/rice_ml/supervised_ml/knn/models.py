"""
User-friendly k-nearest neighbors models.

This module provides three estimators:

- ``KNNClassifier`` for classification
- ``KNNRegressor`` for regression
- ``KNNRecommender`` for basic user-based collaborative filtering

The distance metric may be given as the string ``"euclidean"``,
the string ``"taxicab"``, or as a custom callable.

Examples
--------
>>> X = [[0.0], [1.0], [2.0], [3.0]]
>>> y = [0, 0, 1, 1]
>>> clf = KNNClassifier(n_neighbors=3)
>>> clf.fit(X, y)
KNNClassifier(distance=euclidean, n_neighbors=3, weights=uniform)
>>> clf.predict([[2.1]]).tolist()
[1]

>>> reg = KNNRegressor(n_neighbors=2)
>>> reg.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 4.0])
KNNRegressor(distance=euclidean, n_neighbors=2, weights=uniform)
>>> reg.predict([[1.5]]).tolist()
[2.5]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from rice_ml.measures_ml.distances import euclidean, taxicab


DistanceFunction = Callable[[Sequence[float], Sequence[float]], float]


def _resolve_distance(distance: str | DistanceFunction) -> tuple[str | DistanceFunction, DistanceFunction]:
    """
    Resolve a distance specification into a callable.

    Parameters
    ----------
    distance : str or callable
        Either ``"euclidean"``, ``"taxicab"``, or a callable accepting two
        one-dimensional vectors.

    Returns
    -------
    tuple[str or callable, callable]
        The original specification and the resolved callable.

    Raises
    ------
    TypeError
        If ``distance`` is neither a supported string nor callable.
    ValueError
        If ``distance`` is a string but not a supported metric.
    """
    if callable(distance):
        return distance, distance

    if not isinstance(distance, str):
        raise TypeError("distance must be a string metric name or a callable.")

    metrics = {
        "euclidean": euclidean,
        "taxicab": taxicab,
    }

    if distance not in metrics:
        raise ValueError("Unsupported distance. Use 'euclidean', 'taxicab', or a callable.")

    return distance, metrics[distance]


def _validate_feature_matrix(X: Sequence[Sequence[float]], *, name: str = "X") -> np.ndarray:
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

    Raises
    ------
    TypeError
        If the data cannot be converted to numeric values.
    ValueError
        If the input is empty, non-finite, or not two-dimensional.
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


def _validate_target_vector(y: Sequence, n_samples: int, *, name: str = "y") -> np.ndarray:
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

    Raises
    ------
    ValueError
        If the input length does not match the number of rows in ``X`` or if
        the array is not one-dimensional.
    """
    y_arr = np.asarray(y)

    if y_arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if y_arr.shape[0] != n_samples:
        raise ValueError(f"{name} must have the same number of rows as X.")

    return y_arr


@dataclass
class _BaseKNN:
    """
    Base functionality for KNN estimators.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors to use.
    distance : str or callable, default="euclidean"
        Distance metric specification.
    weights : {"uniform", "distance"}, default="uniform"
        Weighting scheme.
    """

    n_neighbors: int = 5
    distance: str | DistanceFunction = "euclidean"
    weights: str = "uniform"

    def __post_init__(self) -> None:
        if not isinstance(self.n_neighbors, int) or self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")

        if self.weights not in {"uniform", "distance"}:
            raise ValueError("weights must be either 'uniform' or 'distance'.")

        self._distance_spec, self._distance_func = _resolve_distance(self.distance)
        self._is_fitted = False

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """
        Compute distances from a query point to all training samples.

        Parameters
        ----------
        x : numpy.ndarray
            One query sample.

        Returns
        -------
        numpy.ndarray
            Distances to all training points.
        """
        return np.array([self._distance_func(train_x, x) for train_x in self._X], dtype=float)

    def _neighbor_info(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute neighbor indices and distances for a query point.

        Parameters
        ----------
        x : numpy.ndarray
            Query sample.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Neighbor indices and distances.

        Raises
        ------
        ValueError
            If the model has not yet been fitted.
        """
        if not self._is_fitted:
            raise ValueError("This model must be fitted before prediction.")

        distances = self._compute_distances(x)
        order = np.argsort(distances, kind="stable")
        idx = order[: self.n_neighbors]
        return idx, distances[idx]

    def _weights_from_distances(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute sample weights from neighbor distances.

        Parameters
        ----------
        distances : numpy.ndarray
            Distances of selected neighbors.

        Returns
        -------
        numpy.ndarray
            Neighbor weights.
        """
        if self.weights == "uniform":
            return np.ones_like(distances, dtype=float)

        if np.any(distances == 0):
            return (distances == 0).astype(float)

        return 1.0 / distances


class KNNClassifier(_BaseKNN):
    """
    K-nearest neighbors classifier.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors to use.
    distance : str or callable, default="euclidean"
        Distance metric specification.
    weights : {"uniform", "distance"}, default="uniform"
        Weighting scheme.

    Examples
    --------
    >>> X = [[0.0], [1.0], [2.0], [3.0]]
    >>> y = [0, 0, 1, 1]
    >>> model = KNNClassifier(n_neighbors=3)
    >>> model.fit(X, y)
    KNNClassifier(distance=euclidean, n_neighbors=3, weights=uniform)
    >>> model.predict([[1.1]]).tolist()
    [0]
    """

    def fit(self, X: Sequence[Sequence[float]], y: Sequence) -> "KNNClassifier":
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
        KNNClassifier
            Fitted classifier.

        Raises
        ------
        ValueError
            If inputs have incompatible shapes or if ``n_neighbors`` exceeds
            the number of training samples.
        """
        self._X = _validate_feature_matrix(X, name="X")
        self._y = _validate_target_vector(y, self._X.shape[0], name="y")

        if self.n_neighbors > self._X.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples.")

        self.classes_ = np.unique(self._y)
        self._is_fitted = True
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Predict class labels.

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
        if X_query.shape[1] != self._X.shape[1]:
            raise ValueError("X must have the same number of features as the training data.")

        predictions = []
        for row in X_query:
            idx, distances = self._neighbor_info(row)
            labels = self._y[idx]
            weights = self._weights_from_distances(distances)

            scores = {}
            first_seen = {}
            for pos, (label, weight) in enumerate(zip(labels, weights)):
                scores[label] = scores.get(label, 0.0) + float(weight)
                if label not in first_seen:
                    first_seen[label] = pos

            best_label = max(scores, key=lambda label: (scores[label], -first_seen[label]))
            predictions.append(best_label)

        return np.asarray(predictions)


class KNNRegressor(_BaseKNN):
    """
    K-nearest neighbors regressor.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors to use.
    distance : str or callable, default="euclidean"
        Distance metric specification.
    weights : {"uniform", "distance"}, default="uniform"
        Weighting scheme.

    Examples
    --------
    >>> model = KNNRegressor(n_neighbors=2)
    >>> model.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 4.0])
    KNNRegressor(distance=euclidean, n_neighbors=2, weights=uniform)
    >>> model.predict([[1.5]]).tolist()
    [2.5]
    """

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> "KNNRegressor":
        """
        Fit the regressor.

        Parameters
        ----------
        X : sequence of sequence of float
            Training features.
        y : sequence of float
            Numeric training targets.

        Returns
        -------
        KNNRegressor
            Fitted regressor.
        """
        self._X = _validate_feature_matrix(X, name="X")
        y_arr = _validate_target_vector(y, self._X.shape[0], name="y")

        try:
            self._y = y_arr.astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError("y must contain numeric values for regression.") from exc

        if not np.all(np.isfinite(self._y)):
            raise ValueError("y must contain only finite numeric values.")

        if self.n_neighbors > self._X.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples.")

        self._is_fitted = True
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Predict numeric responses.

        Parameters
        ----------
        X : sequence of sequence of float
            Query feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted numeric responses.
        """
        if not self._is_fitted:
            raise ValueError("This model must be fitted before prediction.")

        X_query = _validate_feature_matrix(X, name="X")
        if X_query.shape[1] != self._X.shape[1]:
            raise ValueError("X must have the same number of features as the training data.")

        predictions = []
        for row in X_query:
            idx, distances = self._neighbor_info(row)
            targets = self._y[idx]
            weights = self._weights_from_distances(distances)

            if weights.sum() == 0:
                predictions.append(float(np.mean(targets)))
            else:
                predictions.append(float(np.average(targets, weights=weights)))

        return np.asarray(predictions, dtype=float)


class KNNRecommender(_BaseKNN):
    """
    User-based KNN recommender using a user-item ratings matrix.

    Missing ratings should be encoded as ``numpy.nan``.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors to use.
    distance : str or callable, default="euclidean"
        Distance metric specification.
    weights : {"uniform", "distance"}, default="uniform"
        Weighting scheme.

    Notes
    -----
    This recommender identifies similar users based on overlapping rated items,
    then predicts unseen item ratings by averaging neighbor ratings.

    Examples
    --------
    >>> ratings = [
    ...     [5.0, 4.0, np.nan],
    ...     [4.0, np.nan, 5.0],
    ...     [1.0, 2.0, 1.0],
    ... ]
    >>> model = KNNRecommender(n_neighbors=2)
    >>> model.fit(ratings)
    KNNRecommender(distance=euclidean, n_neighbors=2, weights=uniform)
    >>> recs = model.recommend(user_index=0, top_n=1)
    >>> isinstance(recs, list)
    True
    """

    def fit(self, ratings: Sequence[Sequence[float]]) -> "KNNRecommender":
        """
        Fit the recommender from a user-item rating matrix.

        Parameters
        ----------
        ratings : sequence of sequence of float
            User-item matrix with missing values allowed as ``numpy.nan``.

        Returns
        -------
        KNNRecommender
            Fitted recommender.
        """
        try:
            R = np.asarray(ratings, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("ratings must be a numeric 2D array-like structure.") from exc

        if R.ndim != 2:
            raise ValueError("ratings must be a 2D user-item matrix.")

        if R.shape[0] == 0 or R.shape[1] == 0:
            raise ValueError("ratings must be non-empty.")

        if self.n_neighbors > R.shape[0] - 1 and R.shape[0] > 1:
            raise ValueError("n_neighbors is too large for the number of available other users.")

        self._ratings = R
        self._is_fitted = True
        return self

    def _pair_distance(self, user_a: int, user_b: int) -> float:
        """
        Compute distance between two users on co-rated items.

        Parameters
        ----------
        user_a : int
            Index of first user.
        user_b : int
            Index of second user.

        Returns
        -------
        float
            Distance between user vectors on overlapping items. Returns
            ``np.inf`` if no overlapping ratings exist.
        """
        a = self._ratings[user_a]
        b = self._ratings[user_b]
        mask = np.isfinite(a) & np.isfinite(b)

        if not np.any(mask):
            return float(np.inf)

        return float(self._distance_func(a[mask], b[mask]))

    def predict_rating(self, user_index: int, item_index: int) -> float:
        """
        Predict a rating for one user-item pair.

        Parameters
        ----------
        user_index : int
            User row index.
        item_index : int
            Item column index.

        Returns
        -------
        float
            Predicted rating.

        Raises
        ------
        ValueError
            If the model is unfitted or if no usable neighbors exist.
        IndexError
            If the requested user or item index is out of bounds.
        """
        if not self._is_fitted:
            raise ValueError("This model must be fitted before prediction.")

        n_users, n_items = self._ratings.shape
        if not 0 <= user_index < n_users:
            raise IndexError("user_index is out of bounds.")
        if not 0 <= item_index < n_items:
            raise IndexError("item_index is out of bounds.")

        if np.isfinite(self._ratings[user_index, item_index]):
            return float(self._ratings[user_index, item_index])

        neighbor_data = []
        for other_user in range(n_users):
            if other_user == user_index:
                continue

            candidate_rating = self._ratings[other_user, item_index]
            if not np.isfinite(candidate_rating):
                continue

            dist = self._pair_distance(user_index, other_user)
            if np.isfinite(dist):
                neighbor_data.append((other_user, dist, candidate_rating))

        if not neighbor_data:
            item_mean = np.nanmean(self._ratings[:, item_index])
            if np.isfinite(item_mean):
                return float(item_mean)
            raise ValueError("No usable neighbors or item ratings are available for prediction.")

        neighbor_data.sort(key=lambda row: row[1])
        neighbor_data = neighbor_data[: self.n_neighbors]

        distances = np.array([row[1] for row in neighbor_data], dtype=float)
        ratings = np.array([row[2] for row in neighbor_data], dtype=float)
        weights = self._weights_from_distances(distances)

        if weights.sum() == 0:
            return float(np.mean(ratings))

        return float(np.average(ratings, weights=weights))

    def recommend(self, user_index: int, top_n: int = 5) -> list[tuple[int, float]]:
        """
        Recommend unrated items for one user.

        Parameters
        ----------
        user_index : int
            User row index.
        top_n : int, default=5
            Number of recommendations to return.

        Returns
        -------
        list of tuple[int, float]
            List of ``(item_index, predicted_rating)`` pairs sorted by predicted
            rating in descending order.

        Raises
        ------
        ValueError
            If ``top_n`` is not positive or the model is unfitted.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")

        if not self._is_fitted:
            raise ValueError("This model must be fitted before recommendation.")

        n_users, n_items = self._ratings.shape
        if not 0 <= user_index < n_users:
            raise IndexError("user_index is out of bounds.")

        unrated_items = np.where(~np.isfinite(self._ratings[user_index]))[0]
        scored = []

        for item_index in unrated_items:
            try:
                pred = self.predict_rating(user_index, item_index)
                scored.append((int(item_index), float(pred)))
            except ValueError:
                continue

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n]

    def recommend_frame(self, user_index: int, top_n: int = 5) -> pd.DataFrame:
        """
        Return recommendations as a pandas DataFrame.

        Parameters
        ----------
        user_index : int
            User row index.
        top_n : int, default=5
            Number of recommendations to return.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``item_index`` and ``predicted_rating``.
        """
        recs = self.recommend(user_index=user_index, top_n=top_n)
        return pd.DataFrame(recs, columns=["item_index", "predicted_rating"])
