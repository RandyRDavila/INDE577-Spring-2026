"""
Supervised machine learning algorithms for ``rice_ml``.
"""

from .decision_tree import DecisionTreeClassifier
from .ensemble import AdaBoostClassifier, RandomForestClassifier
from .knn import KNNClassifier, KNNRegressor, KNNRecommender

__all__ = [
    "AdaBoostClassifier",
    "DecisionTreeClassifier",
    "KNNClassifier",
    "KNNRegressor",
    "KNNRecommender",
    "RandomForestClassifier",
]
