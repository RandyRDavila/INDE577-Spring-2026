"""
k-nearest neighbors models for ``rice_ml``.

This submodule provides user-friendly classes for KNN classification,
regression, and recommendation.

Examples
--------
>>> from rice_ml.supervised_ml.knn import KNNClassifier
>>> X = [[0.0], [1.0], [2.0], [3.0]]
>>> y = [0, 0, 1, 1]
>>> model = KNNClassifier(n_neighbors=3)
>>> model.fit(X, y)
KNNClassifier(distance='euclidean', n_neighbors=3, weights='uniform')
>>> model.predict([[1.1]]).tolist()
[0]
"""

from .models import KNNClassifier, KNNRegressor, KNNRecommender

__all__ = ["KNNClassifier", "KNNRegressor", "KNNRecommender"]
