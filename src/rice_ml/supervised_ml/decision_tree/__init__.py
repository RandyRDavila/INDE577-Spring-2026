"""
Decision tree models for ``rice_ml``.

This submodule currently provides a user-friendly classifier for
supervised learning tasks.

Examples
--------
>>> from rice_ml.supervised_ml.decision_tree import DecisionTreeClassifier
>>> X = [[0.0], [1.0], [2.0], [3.0]]
>>> y = [0, 0, 1, 1]
>>> model = DecisionTreeClassifier(max_depth=1)
>>> _ = model.fit(X, y)
>>> model.predict([[2.2]]).tolist()
[1]
"""

from .models import DecisionTreeClassifier

__all__ = ["DecisionTreeClassifier"]
